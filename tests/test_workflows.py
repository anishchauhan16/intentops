"""Exercise data/report/benchmark workflows using synthetic inputs only."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from intentops.config import Settings
from intentops.data import sha256_file, split_development
from intentops.model import Baseline
from intentops.schemas import SemanticMatch
from intentops.semantic import SemanticIndex, SemanticResult
from scripts import evaluate as evaluate_cli
from scripts import fetch_data
from scripts.benchmark_api import benchmark
from scripts.evaluate import evaluate


def mock_download(monkeypatch: pytest.MonkeyPatch) -> Mock:
    names = [f"intent_{i}" for i in range(77)]
    metadata = {"default": {"features": {"label": {"names": names}}}}
    monkeypatch.setattr(
        fetch_data, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(metadata).encode())
    )
    frames = {
        split: pd.DataFrame(
            {
                "text": [f"synthetic example {i}" for i in range(size)],
                "category": [i % 77 for i in range(size)],
            }
        )
        for split, size in (("train", 10003), ("test", 3080))
    }
    loader = Mock(
        return_value={
            split: SimpleNamespace(to_pandas=Mock(return_value=frame))
            for split, frame in frames.items()
        }
    )
    # CI deliberately does not install the optional datasets dependency.
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            ClassLabel=lambda **kwargs: kwargs,
            Features=lambda value: value,
            Value=lambda value: value,
            load_dataset=loader,
        ),
    )
    return loader


def test_fetch_preserves_rows_labels_checksums_and_determinism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = mock_download(monkeypatch)
    first = fetch_data.fetch_data(tmp_path)
    assert first["dataset_name"] == "PolyAI/banking77"
    assert loader.call_args.args == ("csv",)
    for split, size in (("train", 10003), ("test", 3080)):
        frame = pd.read_parquet(tmp_path / f"{split}.parquet")
        assert len(frame) == size
        assert frame.text.tolist() == [f"synthetic example {i}" for i in range(size)]
        assert frame.intent.tolist() == [f"intent_{i % 77}" for i in range(size)]
        assert frame.label.tolist() == [i % 77 for i in range(size)]
        assert first["splits"][split]["sha256"] == sha256_file(tmp_path / f"{split}.parquet")
    assert json.loads((tmp_path / "manifest.json").read_text()) == first
    assert fetch_data.fetch_data(tmp_path) == first


def test_fetch_failure_is_actionable_and_preserves_previous_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = mock_download(monkeypatch)
    fetch_data.fetch_data(tmp_path)
    previous = (tmp_path / "test.parquet").read_bytes()
    loader.side_effect = OSError("simulated download failure")
    with pytest.raises(RuntimeError, match="download/serialization failed.*simulated"):
        fetch_data.fetch_data(tmp_path)
    assert (tmp_path / "test.parquet").read_bytes() == previous
    loader.side_effect = None
    loader.return_value["test"].to_pandas.return_value = pd.DataFrame({"text": ["broken"]})
    with pytest.raises(RuntimeError, match="download/serialization failed"):
        fetch_data.fetch_data(tmp_path)
    assert (tmp_path / "test.parquet").read_bytes() == previous


def test_fetch_serialization_failure_is_actionable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_download(monkeypatch)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", Mock(side_effect=OSError("disk full")))
    with pytest.raises(RuntimeError, match="download/serialization failed.*disk full"):
        fetch_data.fetch_data(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_evaluation_reports_match_known_predictions(
    artifact_dir: Path,
    baseline: Baseline,
    tmp_path: Path,
) -> None:
    # Force one correct and one incorrect example without relying on quality assumptions.
    texts = ["reset my pin", "new card delivery"]
    guesses = [baseline.top_k(text)[0].intent for text in texts]
    other = next(label for label in baseline.labels if label != guesses[1])
    frame = pd.DataFrame({"text": texts, "intent": [guesses[0], other]})
    report = evaluate(frame, Settings(artifact_dir=artifact_dir), tmp_path, "test")
    assert report["baseline"]["accuracy"] == 0.5
    assert report["baseline"]["top_3_accuracy"] == 1
    assert report["hybrid"] is None
    errors = pd.read_csv(tmp_path / "errors.csv")
    assert errors.true_intent.tolist() == [other]
    assert errors.text.tolist() == [texts[1]]
    assert len(json.loads(errors.top_alternatives.iloc[0])) == 3
    assert pd.read_csv(tmp_path / "confusion_pairs.csv")["count"].sum() == 1
    assert len(pd.read_csv(tmp_path / "selective_accuracy.csv")) == 5
    assert json.loads((tmp_path / "metrics.json").read_text()) == report


def test_hybrid_comparison_honestly_reports_a_loss(
    artifact_dir: Path,
    baseline: Baseline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep the session artifact directory unchanged while exercising automatic detection.
    import shutil

    local_artifacts = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, local_artifacts)
    text = "reset my pin"
    correct = baseline.top_k(text)[0].intent
    wrong = next(label for label in baseline.labels if label != correct)
    SemanticIndex(
        np.asarray([[1.0, 0]]), ["public example"], [wrong], "mock", baseline.version
    ).save(local_artifacts / "semantic.npz")
    monkeypatch.setattr(
        SemanticIndex,
        "query",
        lambda self, text: SemanticResult(
            wrong, 1.0, [SemanticMatch(text="public example", intent=wrong, similarity=1.0)]
        ),
    )
    frame = pd.DataFrame({"text": [text], "intent": [correct]})
    output = tmp_path / "reports"
    report = evaluate(
        frame, Settings(artifact_dir=local_artifacts, review_threshold=1), output, "test"
    )
    assert report["baseline"]["accuracy"] == 1
    assert report["hybrid"]["accuracy"] == 0
    assert report["hybrid"]["accuracy_delta_vs_baseline"] == -1
    assert report["hybrid"]["coverage"] == 1
    assert report["hybrid"]["accepted_accuracy"] == 0
    assert report["hybrid"]["semantic_used"] == 1
    assert report["hybrid"]["top_3_accuracy"] is None
    assert pd.read_csv(output / "hybrid_predictions.csv").source.tolist() == ["semantic"]
    (local_artifacts / "semantic.npz").unlink()
    evaluate(frame, Settings(artifact_dir=local_artifacts), output, "test")
    assert not (output / "hybrid_predictions.csv").exists()


def test_benchmark_records_scope_counts_and_real_timings(artifact_dir: Path) -> None:
    result = benchmark(4, 2, artifact_dir)
    assert result["request_count"] == 4
    assert result["warmup_count"] == 2
    assert 0 <= result["p50_ms"] <= result["p95_ms"]
    assert result["measurement"] == "in_process_fastapi_testclient_round_trip"
    assert "network transport" in result["excludes"]
    with pytest.raises(ValueError, match="positive"):
        benchmark(0, 2, artifact_dir)


def test_validation_cli_uses_only_saved_training_split(
    development: pd.DataFrame,
    artifact_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No test.parquet exists: validation must not try to read it.
    development.to_parquet(tmp_path / "train.parquet", index=False)
    output = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--split",
            "validation",
            "--artifact-dir",
            str(artifact_dir),
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
    )
    evaluate_cli.main()
    _, validation = split_development(development)
    report = json.loads((output / "metrics.json").read_text())
    assert report["split"] == "validation"
    assert report["rows"] == len(validation) == 6
