"""Synthetic-only transformer tests: local vocabulary and random tiny weights."""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from intentops.data import sha256_file
from intentops.transformer_evaluation import cpu_timing, evaluate_probabilities, write_comparison
from intentops.transformer_model import (
    TransformerConfig,
    TransformerModel,
    assert_label_order,
    canonical_labels,
    checkpoint_checksum,
    development_data,
    encode_labels,
    seed_everything,
    train_transformer,
)
from scripts import evaluate_transformer
from scripts import train_transformer as training_cli


@pytest.fixture
def labels() -> list[str]:
    return [f"intent_{index:02}" for index in range(77)]


@pytest.fixture
def source(tmp_path: Path, labels: list[str]) -> tuple[Path, Path]:
    data = tmp_path / "data"
    baseline = tmp_path / "baseline"
    data.mkdir()
    baseline.mkdir()
    frame = pd.DataFrame(
        [
            {"text": f"synthetic request {index} example {row}", "intent": label, "label": -1}
            for index, label in enumerate(labels)
            for row in range(129 if index < 7 else 130)
        ]
    )
    frame.to_parquet(data / "train.parquet", index=False)
    manifest = {
        "dataset_name": "synthetic",
        "source_revision": "synthetic",
        "label_revision": "synthetic",
        "intent_names": list(reversed(labels)),
        "splits": {
            "train": {"sha256": sha256_file(data / "train.parquet"), "rows": 10003},
            "test": {"sha256": "test-file-does-not-exist", "rows": 3080},
        },
    }
    (data / "manifest.json").write_text(json.dumps(manifest))
    (baseline / "metadata.json").write_text(json.dumps({"label_names": labels}))
    (baseline / "baseline.joblib").write_bytes(b"synthetic baseline placeholder")
    return data, baseline


@pytest.fixture
def tiny(tmp_path: Path, labels: list[str]) -> TransformerModel:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    seed_everything(threads=1)
    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "synthetic",
                "request",
                "example",
                "hello",
            ]
        )
    )
    tokenizer = transformers.BertTokenizerFast(
        vocab_file=str(vocab), model_input_names=["input_ids", "attention_mask"]
    )
    config = transformers.DistilBertConfig(
        vocab_size=9,
        max_position_embeddings=32,
        n_layers=1,
        n_heads=2,
        dim=8,
        hidden_dim=16,
        num_labels=77,
        id2label=dict(enumerate(labels)),
        label2id={name: index for index, name in enumerate(labels)},
    )
    return TransformerModel(
        transformers.DistilBertForSequenceClassification(config), tokenizer, labels, 16
    )


def test_manifest_integer_order_is_never_used(source: tuple[Path, Path], labels: list[str]) -> None:
    data, baseline = source
    assert canonical_labels(baseline, data / "manifest.json") == labels
    assert encode_labels([labels[76], labels[0]], labels) == [76, 0]
    with pytest.raises(ValueError, match="outside"):
        encode_labels(["unknown"], labels)
    with pytest.raises(ValueError, match="exactly"):
        assert_label_order(list(reversed(labels)), labels)
    with pytest.raises(ValueError, match="exactly"):
        assert_label_order(labels[:-1], labels[:-1])
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["intent_names"][0] = "different-intent"
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same 77"):
        canonical_labels(baseline, data / "manifest.json")


def test_reuses_stratified_split_without_opening_test(source: tuple[Path, Path]) -> None:
    data, baseline = source
    train, validation, labels, provenance = development_data(data, baseline)
    assert (len(train), len(validation), len(labels)) == (8002, 2001, 77)
    assert set(train.index).isdisjoint(validation.index)
    assert provenance["dataset_checksums"]["test"] == "test-file-does-not-exist"
    small_train, small_validation, _, _ = development_data(data, baseline, smoke=True)
    assert len(small_train) == 154 and len(small_validation) == 77
    assert set(small_train.index).issubset(train.index)
    assert set(small_validation.index).issubset(validation.index)
    assert set(small_train.index).isdisjoint(small_validation.index)
    assert small_train["label"].eq(-1).all()
    (data / "train.parquet").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        development_data(data, baseline)


@pytest.mark.parametrize(
    "kwargs", [{"seed": 1}, {"epochs": 0}, {"learning_rate": 0}, {"weight_decay": -1}]
)
def test_invalid_training_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        TransformerConfig(**kwargs)


def test_default_model_is_not_substituted() -> None:
    assert TransformerConfig().model_name == "distilbert-base-uncased"


def test_known_metrics_and_zero_acceptance(tmp_path: Path) -> None:
    frame = pd.DataFrame({"text": ["synthetic"] * 4, "intent": ["a", "b", "c", "a"]})
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.4, 0.3, 0.3], [0.1, 0.6, 0.3]])
    metrics = evaluate_probabilities(frame, probabilities, ["a", "b", "c"], tmp_path / "metrics")
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 3)
    assert metrics["weighted_f1"] == pytest.approx((2 * 0.5 + 2 / 3) / 4)
    assert metrics["top_3_accuracy"] == 1
    assert metrics["gate"]["coverage"] == 0.5
    assert metrics["gate"]["accepted_accuracy"] == 1
    assert len(pd.read_csv(tmp_path / "metrics/errors.csv")) == 2
    assert len(pd.read_csv(tmp_path / "metrics/confusion_pairs.csv")) == 2
    none = evaluate_probabilities(frame, np.full((4, 3), 1 / 3), ["a", "b", "c"], tmp_path / "none")
    assert none["gate"]["accepted_accuracy"] is None


def test_shared_timing_excludes_warmup_and_divides_batch_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(index / 1000 for index in range(12))
    monkeypatch.setattr("intentops.transformer_evaluation.perf_counter", lambda: next(ticks))
    calls: list[int] = []

    def predict(texts: list[str]) -> np.ndarray:
        calls.append(len(texts))
        return np.ones((len(texts), 3)) / 3

    report = cpu_timing(predict, ["a", "b", "c", "d"], (1, 2), 1, 1)
    assert calls == [1] * 5 + [2] * 3
    assert report["batches"][0]["mean_ms_per_example"] == pytest.approx(1)
    assert report["batches"][1]["mean_ms_per_example"] == pytest.approx(0.5)
    assert report["batches"][0]["measured_batches"] == 4
    assert len(report["examples_sha256"]) == 64
    with pytest.raises(ValueError):
        cpu_timing(predict, [], (1,))
    with pytest.raises(ValueError, match="divisible"):
        cpu_timing(predict, ["a"], (2,))


def test_checkpoint_roundtrip_shapes_order_and_tampering(
    tiny: TransformerModel, tmp_path: Path, labels: list[str]
) -> None:
    before = tiny.probabilities(["hello", "synthetic request"], batch_size=1)
    assert before.shape == (2, 77)
    np.testing.assert_allclose(before.sum(axis=1), 1, atol=1e-6)
    assert len(tiny.top_k("hello")) == 3
    with pytest.raises(ValueError):
        tiny.top_k("hello", 0)
    with pytest.raises(ValueError):
        tiny.probabilities([])
    checkpoint = tmp_path / "checkpoint"
    record = tiny.save(checkpoint, {"smoke": True})
    assert record["model_sha256"] == checkpoint_checksum(record["checkpoint_files"])
    restored = TransformerModel.load(checkpoint, labels)
    np.testing.assert_allclose(
        restored.probabilities(["hello", "synthetic request"]), before, atol=1e-6
    )
    with pytest.raises(ValueError, match="exactly"):
        TransformerModel.load(checkpoint, list(reversed(labels)))
    with pytest.raises(FileExistsError):
        tiny.save(checkpoint, {})
    (checkpoint / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        TransformerModel.load(checkpoint, labels)
    tiny.model.config.label2id = {}
    with pytest.raises(ValueError, match="label2id"):
        TransformerModel(tiny.model, tiny.tokenizer, labels, 16)


def test_tiny_training_selects_validation_epoch(
    tiny: TransformerModel, labels: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    train = pd.DataFrame({"text": ["synthetic request"] * 77, "intent": labels})
    validation = train.copy()
    monkeypatch.setattr(
        "intentops.transformer_model.classification_metrics",
        lambda *args: {"macro_f1": 0.1, "accuracy": 0.1},
    )
    record = train_transformer(
        tiny, train, validation, TransformerConfig(epochs=5, batch_size=77, max_length=16)
    )
    assert record["epochs_completed"] == 2
    assert record["selected_epoch"] == 1
    assert record["training_seconds"] > 0
    assert record["configuration"]["seed"] == 42


def test_pretrained_builder_uses_explicit_default_and_mapping(
    tiny: TransformerModel, labels: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    auto_model, auto_tokenizer = Mock(), Mock()
    auto_model.from_pretrained.return_value = tiny.model
    auto_tokenizer.from_pretrained.return_value = tiny.tokenizer
    monkeypatch.setattr(
        "intentops.transformer_model.dependencies", lambda: (None, auto_model, auto_tokenizer)
    )
    built = TransformerModel.build(TransformerConfig(), labels, "cpu", offline=True)
    assert built.labels == labels
    assert auto_model.from_pretrained.call_args.args == ("distilbert-base-uncased",)
    assert auto_model.from_pretrained.call_args.kwargs["local_files_only"] is True
    assert auto_model.from_pretrained.call_args.kwargs["id2label"] == dict(enumerate(labels))


def test_final_test_is_frozen_one_shot_and_forbidden_for_smoke(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    configuration = {"checkpoint": "synthetic", "threshold": 0.7}
    freeze.write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="Smoke"):
        evaluate_transformer.claim_final_test(tmp_path, freeze, configuration, True)
    with pytest.raises(ValueError, match="Frozen"):
        evaluate_transformer.claim_final_test(tmp_path, freeze, {}, False)
    receipt = evaluate_transformer.claim_final_test(tmp_path, freeze, configuration, False)
    assert json.loads(receipt.read_text())["status"] == "started"
    with pytest.raises(FileExistsError):
        evaluate_transformer.claim_final_test(tmp_path, freeze, configuration, False)


def test_evaluation_cli_rejects_unfrozen_test_before_any_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_transformer.py", "--split", "test"])
    with pytest.raises(SystemExit, match="1"):
        evaluate_transformer.main()


def test_train_cli_smoke_uses_only_development(
    tiny: TransformerModel,
    source: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, baseline = source
    output = tmp_path / "trained"
    monkeypatch.setattr(TransformerModel, "build", lambda *args: tiny)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_transformer.py",
            "--data-dir",
            str(data),
            "--baseline-dir",
            str(baseline),
            "--artifact-dir",
            str(output),
            "--smoke",
        ],
    )
    training_cli.main()
    record = json.loads((output / "metadata.json").read_text())
    assert record["training_rows"] == 154 and record["validation_rows"] == 77
    assert record["smoke"] is True and record["hardware"]["device"] == "cpu"
    assert "synthetic request" not in capsys.readouterr().out


def test_comparison_keeps_smoke_results_pending(tmp_path: Path) -> None:
    metrics = {
        "accuracy": 0.5,
        "macro_f1": 0.4,
        "weighted_f1": 0.4,
        "top_3_accuracy": 0.8,
        "gate": {"coverage": 0.2, "accepted_accuracy": 0.9},
    }
    entry = {
        "metrics": metrics,
        "artifact_bytes": 123,
        "latency": {
            "batches": [{"batch_size": 1, "mean_ms_per_example": 1.0, "p95_ms_per_example": 2.0}]
        },
        "training_seconds": 5.0,
        "training_hardware": {"device": "cpu"},
    }
    for smoke in (True, False):
        report = {
            "smoke": smoke,
            "split": "validation",
            "baseline": entry,
            "transformer": entry,
            "timing_configuration": {"batch_sizes": [1]},
        }
        output = tmp_path / str(smoke)
        write_comparison(report, output)
        saved = json.loads(output.with_suffix(".json").read_text())
        assert saved["comparison"]["baseline"]["training_seconds"] is None
        assert saved["comparison"]["transformer"]["accuracy"] == (None if smoke else 0.5)
        assert saved["run_evidence"]["transformer"]["metrics"]["accuracy"] == 0.5


def test_validation_freeze_then_exactly_one_synthetic_final_evaluation(
    tiny: TransformerModel,
    source: tuple[Path, Path],
    tmp_path: Path,
    labels: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from intentops.data import read_data

    data, baseline = source
    # This is generated test-fixture data, not the official BANKING77 test file.
    pd.DataFrame(
        [{"text": "synthetic request", "intent": label} for label in labels for _ in range(40)]
    ).to_parquet(data / "test.parquet", index=False)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["splits"]["test"]["sha256"] = sha256_file(data / "test.parquet")
    (data / "manifest.json").write_text(json.dumps(manifest))
    _, _, _, provenance = development_data(data, baseline)
    artifact = tmp_path / "complete-checkpoint"
    tiny.save(
        artifact,
        {**provenance, "smoke": False, "training_seconds": 1.0, "hardware": {"device": "cpu"}},
    )
    fake_baseline = SimpleNamespace(
        labels=labels,
        probabilities=lambda texts: np.full((len(texts), 77), 1 / 77),
    )
    monkeypatch.setattr(evaluate_transformer.Baseline, "load", lambda *args: fake_baseline)
    opened_test = []

    def tracked_read(path: Path) -> pd.DataFrame:
        opened_test.append(path)
        return read_data(path)

    monkeypatch.setattr(evaluate_transformer, "read_data", tracked_read)
    common = [
        "evaluate_transformer.py",
        "--data-dir",
        str(data),
        "--baseline-dir",
        str(baseline),
        "--artifact-dir",
        str(artifact),
        "--output-dir",
        str(tmp_path / "reports"),
        "--comparison",
        str(tmp_path / "comparison"),
        "--timing-examples",
        "4",
        "--batch-size",
        "2",
        "--warmup",
        "1",
        "--repeats",
        "1",
    ]
    freeze = tmp_path / "freeze.json"
    monkeypatch.setattr(sys, "argv", common + ["--freeze", str(freeze)])
    evaluate_transformer.main()
    assert opened_test == []
    assert freeze.is_file()
    final = common + ["--split", "test", "--final-test", "--freeze-file", str(freeze)]
    monkeypatch.setattr(sys, "argv", final)
    evaluate_transformer.main()
    assert opened_test == [data / "test.parquet"]
    assert (
        json.loads((artifact / "test_evaluation_receipt.json").read_text())["status"] == "completed"
    )
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["split"] == "test" and report["run_evidence"]["rows"] == 3080
    with pytest.raises(SystemExit, match="1"):
        evaluate_transformer.main()
    assert opened_test == [data / "test.parquet"]
