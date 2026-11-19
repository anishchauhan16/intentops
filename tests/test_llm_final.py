"""Synthetic-only selection and freeze guards; never opens the official dataset."""

import inspect
import json
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from intentops.data import sha256_file
from intentops.llm_cache import ResponseCache, atomic_json, fingerprint
from intentops.llm_evaluation import quality
from intentops.llm_final import (
    CacheOnly,
    claim_test,
    create_freeze,
    gate_metrics,
    prompt_contract,
    read_final_test,
    save_comparison,
    select_tau,
    source_identity,
    summarize_run,
)
from intentops.llm_path import LocalConfig, build_graph

LABELS = [f"synthetic_{i}" for i in range(77)]


def row(correct: bool = True, similarity: float = 0.9) -> tuple[dict[str, Any], str]:
    return (
        {
            "intent": LABELS[0],
            "baseline_label": LABELS[0],
            "nearest_label": LABELS[0],
            "nearest_similarity": similarity,
        },
        LABELS[0] if correct else LABELS[1],
    )


def select(examples: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    return select_tau([value[0] for value in examples], [value[1] for value in examples])


def test_eligible_tau_maximizes_coverage_and_preserves_similarity_ties() -> None:
    result = select([row(True)] * 500 + [row(False)] * 5 + [row(False, 0.2)] * 100)
    assert result["target_status"] == "MET"
    chosen = result["chosen"]
    assert chosen["tau"] == 0.2 and chosen["support"] == 505 and chosen["correct"] == 500
    assert chosen["coverage"] == pytest.approx(505 / 605)
    assert chosen["wilson_95"][0] < chosen["accepted_accuracy"] < chosen["wilson_95"][1]
    assert all(value["support"] in {0, 505, 605} for value in result["sweep"])


def test_exact_99_percent_and_400_support_is_eligible() -> None:
    result = select([row(True)] * 396 + [row(False)] * 4)
    assert result["target_status"] == "MET" and result["chosen"]["support"] == 400
    assert result["chosen"]["accepted_accuracy"] == 0.99


def test_fallback_does_not_relax_minimum_support_and_uses_declared_ties() -> None:
    result = select([row(True)] * 399 + [row(False, 0.1)] * 6)
    assert result["target_status"] == "MISSED" and result["eligible_thresholds"] == 0
    assert result["chosen"]["support"] == 399 and result["chosen"]["accepted_accuracy"] == 1
    tied = select([row(True)] * 300 + [row(True, 0.8)] * 20 + [row(False, 0.7)])
    assert tied["target_status"] == "MISSED"
    assert tied["chosen"]["tau"] == 0.7 and tied["chosen"]["support"] == 320


def test_invalid_and_disagreement_rows_never_enter_sweep() -> None:
    invalid = row()[0]
    invalid["intent"] = None
    mismatch = row()[0]
    mismatch["baseline_label"] = LABELS[1]
    result = select([(invalid, LABELS[0]), (mismatch, LABELS[0])])
    assert result["target_status"] == "MISSED"
    assert result["chosen"]["support"] == 0 and result["chosen"]["accepted_accuracy"] is None
    with pytest.raises(ValueError):
        select([row(True, float("nan"))])
    with pytest.raises(ValueError):
        gate_metrics([], [], 0.5)


def test_prompt_contract_ignores_tau_but_detects_prompt_or_label_change() -> None:
    import intentops.llm_path

    config = LocalConfig()
    expected = prompt_contract(config, LABELS)
    assert prompt_contract(config, LABELS, inspect.getsource(intentops.llm_path)) == expected
    assert prompt_contract(replace(config, tau=0.8), LABELS) == expected
    assert prompt_contract(config, list(reversed(LABELS))) != expected
    source = inspect.getsource(build_graph)
    changed = source.replace("Do not ask questions.", "Ask questions.")
    assert prompt_contract(config, LABELS, changed)["prompt_sha256"] != expected["prompt_sha256"]
    with pytest.raises((ValueError, StopIteration)):
        prompt_contract(config, LABELS, "def classify(state):\n    return {}")


def test_cache_only_never_calls_provider_on_miss(tmp_path: Path) -> None:
    pytest.importorskip("filelock")
    key = fingerprint("synthetic")
    cache = CacheOnly(tmp_path)

    def forbidden() -> dict[str, Any]:
        pytest.fail("Cache-only replay must not call a provider.")

    with pytest.raises(ValueError, match="Pilot cache miss"):
        cache.resolve(key, forbidden)
    ResponseCache(tmp_path).resolve(key, lambda: {"intent": LABELS[0]})
    result, hit = cache.resolve(key, forbidden)
    assert hit and result == {"intent": LABELS[0]}


def test_freeze_and_exclusive_claim_reject_changes_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = {"code_commit": "synthetic-commit", "source_sha256": {"synthetic.py": "hash"}}
    monkeypatch.setattr("intentops.llm_final.source_identity", lambda: identity)
    config = LocalConfig()
    validation = tmp_path / "validation.json"
    atomic_json(validation, {"synthetic": True})
    selection = select([row(True)] * 400)
    freeze = tmp_path / "freeze.json"
    models = {"baseline": "synthetic"}
    frozen = create_freeze(freeze, config, LABELS, {}, selection, validation, models)
    assert frozen["code_commit"] == identity["code_commit"]
    assert frozen["configuration"]["tau"] == selection["chosen"]["tau"]
    with pytest.raises(FileExistsError):
        create_freeze(freeze, config, LABELS, {}, selection, validation, models)
    attempt = tmp_path / "attempt.json"
    claim_test(attempt, freeze, config, LABELS, {}, validation, models)
    assert json.loads(attempt.read_text())["status"] == "started"
    with pytest.raises(FileExistsError):
        claim_test(attempt, freeze, config, LABELS, {}, validation, models)
    invalid_attempt = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Freeze receipt"):
        claim_test(
            invalid_attempt,
            freeze,
            replace(config, prompt_version="different"),
            LABELS,
            {},
            validation,
            models,
        )
    assert not invalid_attempt.exists()
    identity["code_commit"] = "modified-code"
    with pytest.raises(ValueError, match="Freeze receipt"):
        claim_test(invalid_attempt, freeze, config, LABELS, {}, validation, models)


def test_single_test_file_read_with_synthetic_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"text": ["synthetic support"] * 3080, "intent": LABELS * 40})
    path = tmp_path / "synthetic.parquet"
    frame.to_parquet(path)
    checksum = sha256_file(path)
    reads = []
    original = Path.read_bytes

    def tracked(self: Path) -> bytes:
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    loaded = read_final_test(path, checksum, LABELS)
    assert len(loaded) == 3080 and reads == [path]
    with pytest.raises(ValueError, match="checksum"):
        read_final_test(path, "wrong", LABELS)
    frame.iloc[:10].to_parquet(path)
    with pytest.raises(ValueError, match="every official"):
        read_final_test(path, sha256_file(path), LABELS)


def test_source_identity_refuses_uncommitted_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intentops.llm_final.subprocess.check_output", lambda *a, **kw: " M source.py"
    )
    with pytest.raises(ValueError, match="Commit"):
        source_identity()
    monkeypatch.setattr(
        "intentops.llm_final.subprocess.check_output",
        lambda args, **kw: "" if "status" in args else "synthetic-commit",
    )
    monkeypatch.setattr("intentops.llm_final.sha256_file", lambda path: "synthetic-hash")
    assert source_identity()["code_commit"] == "synthetic-commit"


def test_final_three_way_report_uses_test_labels_and_frozen_gate(tmp_path: Path) -> None:
    frame = pd.DataFrame({"text": ["synthetic"], "intent": [LABELS[0]]})
    example = row()[0]
    example.update(
        {
            "row_id": 0,
            "row_sha256": fingerprint("synthetic"),
            "failure": None,
            "cache_hit": False,
            "prompt_eval_count": 10,
            "eval_count": 2,
            "prompt_eval_cached_count": 0,
            **{f"{stage}_seconds": 0.1 for stage in ("retrieval", "baseline", "llm", "end_to_end")},
        }
    )
    metrics = quality([LABELS[0]], [LABELS[0]], LABELS)
    report = summarize_run(
        "test",
        frame,
        [example],
        0.2,
        {name: metrics for name in ("baseline", "transformer", "llm_rag")},
        0.8,
    )
    assert report["gate"]["support"] == 1
    report["reproduction"] = {"baseline": {"matches": True}, "transformer": {"matches": True}}
    document = {"test": report, "validation": report, "selection": {"target_status": "MISSED"}}
    save_comparison(tmp_path / "comparison", document)
    markdown = (tmp_path / "comparison.md").read_text()
    assert "Macro F1" in markdown and "Weighted F1" in markdown and "MISSED" in markdown
    assert "Single final test" in markdown and "$0.00" in markdown
    with pytest.raises(ValueError, match="ordered"):
        summarize_run("test", frame, [{**example, "row_id": 9}], 0.2, {}, 0.8)


def test_final_runner_synthetic_end_to_end_and_second_attempt_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/finalize_llm.py"))
    main = namespace["main"]
    scope = main.__globals__
    monkeypatch.chdir(tmp_path)
    config = LocalConfig()
    validation = pd.DataFrame(
        {
            "text": [f"synthetic_{i}" for i in range(2001)],
            "intent": [LABELS[i % 77] for i in range(2001)],
        }
    )
    pilot_frame = validation.iloc[:250]
    binding = {"index": "synthetic"}
    provenance = {
        "baseline_sha256": "baseline-sha",
        "manifest_sha256": "manifest-sha",
        "dataset_checksums": {"test": "expected-synthetic-checksum"},
    }
    atomic_json(Path("configs/llm-pilot.json"), asdict(config))
    atomic_json(
        Path("artifacts/transformer/metadata.json"),
        {
            "model_version": "transformer",
            "model_sha256": "transformer-sha",
            "smoke": False,
            "baseline_sha256": "baseline-sha",
            "manifest_sha256": "manifest-sha",
        },
    )
    atomic_json(
        Path("reports/phase2_pilot.json"),
        {
            "configuration": asdict(config),
            "runtime": {},
            "baseline_version": "baseline",
            "transformer_version": "transformer",
            "index_binding": binding,
            "hardware": {"ollama_version": "synthetic"},
            "row_ids": pilot_frame.index.tolist(),
            "row_hashes": [
                fingerprint([int(i), value.text, value.intent])
                for i, value in pilot_frame.iterrows()
            ],
        },
    )
    atomic_json(
        Path("reports/transformer_vs_baseline.json"),
        {"comparison": {name: {"accuracy": 1.0} for name in ("baseline", "transformer")}},
    )
    identity = {"code_commit": "synthetic-commit", "source_sha256": {}}
    monkeypatch.setattr("intentops.llm_final.source_identity", lambda: identity)
    monkeypatch.setitem(scope, "source_identity", lambda: identity)
    monkeypatch.setitem(scope, "seed_everything", lambda **kw: None)
    monkeypatch.setitem(
        scope, "development_data", lambda *a: (validation, validation, LABELS, provenance)
    )
    monkeypatch.setitem(scope, "index_binding", lambda *a: binding)
    monkeypatch.setitem(scope, "LocalEncoder", lambda *a: None)
    monkeypatch.setitem(scope, "ChromaRetriever", lambda *a: None)
    monkeypatch.setitem(
        scope,
        "OllamaProvider",
        lambda *a: SimpleNamespace(version="synthetic", verify=lambda: None),
    )
    monkeypatch.setitem(
        scope, "Baseline", SimpleNamespace(load=lambda *a: SimpleNamespace(version="baseline"))
    )

    def probabilities(texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), 77))
        for i, text in enumerate(texts):
            result[i, int(text.split("_")[-1]) % 77] = 1
        return result

    monkeypatch.setitem(
        scope,
        "TransformerModel",
        SimpleNamespace(load=lambda *a: SimpleNamespace(probabilities=probabilities)),
    )
    monkeypatch.setitem(scope, "build_graph", lambda *a: a[-1])
    monkeypatch.setitem(
        scope,
        "subprocess",
        SimpleNamespace(check_output=lambda *a, **kw: inspect.getsource(build_graph)),
    )

    def fake_execute(
        graph: Any, frame: pd.DataFrame, path: Path
    ) -> tuple[list[dict[str, Any]], float]:
        records = []
        for i, value in frame.iterrows():
            records.append(
                {
                    "row_id": int(str(i)),
                    "row_sha256": fingerprint([int(str(i)), value.text, value.intent]),
                    "intent": value.intent,
                    "baseline_label": value.intent,
                    "nearest_label": value.intent,
                    "nearest_similarity": 0.9,
                    "cache_hit": isinstance(graph, CacheOnly),
                    "failure": None,
                    "prompt_eval_count": 10,
                    "eval_count": 2,
                    "prompt_eval_cached_count": 0,
                    **{
                        f"{stage}_seconds": 0.1
                        for stage in ("retrieval", "baseline", "llm", "end_to_end")
                    },
                }
            )
        atomic_json(path, {"rows": records})
        return records, 0.2

    monkeypatch.setitem(scope, "execute", fake_execute)
    monkeypatch.setattr(sys, "argv", ["finalize_llm.py", "--stage", "validation"])
    main()
    assert Path("reports/llm_freeze.json").is_file()
    reads = []

    def fake_test_read(*args: Any) -> pd.DataFrame:
        assert Path("artifacts/llm/final_test_receipt.json").exists()
        reads.append(1)
        return pd.DataFrame(
            {"text": [f"synthetic_{i}" for i in range(3080)], "intent": LABELS * 40}
        )

    monkeypatch.setitem(scope, "read_final_test", fake_test_read)
    monkeypatch.setattr(sys, "argv", ["finalize_llm.py", "--stage", "test"])
    main()
    assert reads == [1]
    final = json.loads(Path("reports/llm_vs_baseline_transformer.json").read_text())
    assert final["validation"]["rows"] == 2001 and final["test"]["rows"] == 3080
    assert final["test"]["reproduction"]["baseline"]["matches"]
    receipt = json.loads(Path("artifacts/llm/final_test_receipt.json").read_text())
    assert receipt["status"] == "completed" and "started_at" in receipt
    with pytest.raises(FileExistsError):
        main()
    assert reads == [1]
