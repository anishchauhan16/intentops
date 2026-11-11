"""Optional transformer training; never imported by the baseline service."""

import hashlib
import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from intentops.config import SEED, VALIDATION_SIZE
from intentops.data import read_data, sha256_file, split_development
from intentops.metrics import classification_metrics
from intentops.schemas import IntentScore


@dataclass(frozen=True)
class TransformerConfig:
    model_name: str = "distilbert-base-uncased"
    revision: str | None = None
    seed: int = SEED
    epochs: int = 3
    batch_size: int = 16
    max_length: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    patience: int = 1
    cpu_threads: int = 2
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.seed != SEED:
            raise ValueError("This comparison requires seed 42.")
        if min(self.epochs, self.batch_size, self.max_length, self.patience, self.cpu_threads) < 1:
            raise ValueError("Training counts must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer configuration.")


def dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            'Install optional training support: pip install -e ".[transformer]"'
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def seed_everything(seed: int = SEED, threads: int = 2) -> dict[str, Any]:
    torch, _, _ = dependencies()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def assert_label_order(labels: list[str], expected: list[str]) -> None:
    if labels != expected or len(labels) != 77 or len(set(labels)) != 77:
        raise ValueError("Transformer label order must exactly match the 77 baseline labels.")


def canonical_labels(baseline_dir: Path, manifest_path: Path) -> list[str]:
    metadata = json.loads((baseline_dir / "metadata.json").read_text())
    labels: list[str] = metadata["label_names"]
    assert_label_order(labels, labels)
    manifest = json.loads(manifest_path.read_text())
    names = manifest["intent_names"]
    if len(names) != 77 or len(set(names)) != 77 or set(names) != set(labels):
        raise ValueError("Dataset manifest and baseline must contain the same 77 intent names.")
    return labels.copy()


def encode_labels(intents: list[str], labels: list[str]) -> list[int]:
    mapping = {name: index for index, name in enumerate(labels)}
    try:
        return [mapping[name] for name in intents]
    except KeyError as exc:
        raise ValueError("Dataset contains an intent outside the baseline mapping.") from exc


def development_data(
    data_dir: Path, baseline_dir: Path, smoke: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Open official TRAIN only; the manifest's test checksum is provenance, not test access."""
    manifest_path = data_dir / "manifest.json"
    labels = canonical_labels(baseline_dir, manifest_path)
    manifest = json.loads(manifest_path.read_text())
    path = data_dir / "train.parquet"
    if sha256_file(path) != manifest["splits"]["train"]["sha256"]:
        raise ValueError("Training data checksum differs from the dataset manifest.")
    frame = read_data(path)
    if len(frame) != 10003 or set(frame["intent"]) != set(labels):
        raise ValueError("Expected the unchanged 10,003-row, 77-intent development split.")
    train, validation = split_development(frame, VALIDATION_SIZE, SEED)
    if smoke:
        train = train.groupby("intent", sort=True).sample(n=2, random_state=SEED)
        validation = validation.groupby("intent", sort=True).sample(n=1, random_state=SEED)
    provenance = {
        "dataset_name": manifest["dataset_name"],
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_checksums": {key: value["sha256"] for key, value in manifest["splits"].items()},
        "source_revision": manifest["source_revision"],
        "label_revision": manifest["label_revision"],
        "official_development_rows": len(frame),
        "development_training_rows": 8002,
        "development_validation_rows": 2001,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "validation_size": VALIDATION_SIZE,
        "baseline_sha256": sha256_file(baseline_dir / "baseline.joblib"),
    }
    return train, validation, labels, provenance


def checkpoint_files(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"metadata.json", "test_evaluation_receipt.json"}
    }


def checkpoint_checksum(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


class TransformerModel:
    def __init__(
        self, model: Any, tokenizer: Any, labels: list[str], max_length: int, device: str = "cpu"
    ) -> None:
        assert_label_order(labels, labels)
        stored = [model.config.id2label[index] for index in range(model.config.num_labels)]
        assert_label_order(stored, labels)
        expected_mapping = {label: index for index, label in enumerate(labels)}
        if model.config.label2id != expected_mapping:
            raise ValueError("Transformer label2id differs from the canonical mapping.")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.labels = labels.copy()
        self.max_length = max_length
        self.device = device

    @classmethod
    def build(
        cls, config: TransformerConfig, labels: list[str], device: str, offline: bool = False
    ) -> "TransformerModel":
        torch, auto_model, auto_tokenizer = dependencies()
        del torch
        assert_label_order(labels, labels)
        options = {
            "revision": config.revision,
            "local_files_only": offline,
            "trust_remote_code": False,
        }
        tokenizer = auto_tokenizer.from_pretrained(config.model_name, **options)
        model = auto_model.from_pretrained(
            config.model_name,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={label: index for index, label in enumerate(labels)},
            use_safetensors=True,
            **options,
        )
        return cls(model, tokenizer, labels, config.max_length, device)

    def inputs(self, texts: list[str]) -> Any:
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)

    def probabilities(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts or batch_size < 1:
            raise ValueError("Prediction requires nonempty texts and a positive batch size.")
        torch, _, _ = dependencies()
        self.model.eval()
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                logits = self.model(**self.inputs(texts[start : start + batch_size])).logits
                batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(batches, axis=0)

    def top_k(self, text: str, k: int = 3) -> list[IntentScore]:
        if k < 1:
            raise ValueError("k must be positive.")
        probabilities = self.probabilities([text])[0]
        indices = np.argsort(-probabilities, kind="stable")[:k]
        return [
            IntentScore(intent=self.labels[index], probability=float(probabilities[index]))
            for index in indices
        ]

    def save(self, directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)
        files = checkpoint_files(directory)
        checksum = checkpoint_checksum(files)
        record = {
            **metadata,
            "label_names": self.labels,
            "max_length": self.max_length,
            "checkpoint_files": files,
            "model_sha256": checksum,
            "model_version": f"transformer-{checksum[:12]}",
        }
        (directory / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    @classmethod
    def load(cls, directory: Path, expected_labels: list[str]) -> "TransformerModel":
        _, auto_model, auto_tokenizer = dependencies()
        record = json.loads((directory / "metadata.json").read_text())
        assert_label_order(record["label_names"], expected_labels)
        files = checkpoint_files(directory)
        if (
            files != record["checkpoint_files"]
            or checkpoint_checksum(files) != record["model_sha256"]
        ):
            raise ValueError("Transformer checkpoint checksum mismatch.")
        model = auto_model.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        tokenizer = auto_tokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
        return cls(model, tokenizer, expected_labels, record["max_length"])


def train_transformer(
    model: TransformerModel,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: TransformerConfig,
) -> dict[str, Any]:
    """Select the best epoch using validation macro F1; never accept test data here."""
    torch, _, _ = dependencies()
    train_ids = encode_labels(train["intent"].tolist(), model.labels)
    validation_truth = validation["intent"].tolist()
    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    best_score, best_epoch, stale = -1.0, 0, 0
    best_state = None
    history = []
    start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.model.train()
        total_loss = 0.0
        order = torch.randperm(len(train), generator=generator).tolist()
        for offset in range(0, len(order), config.batch_size):
            indices = order[offset : offset + config.batch_size]
            inputs = model.inputs(train.iloc[indices]["text"].tolist())
            targets = torch.tensor([train_ids[index] for index in indices], device=model.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.model(**inputs, labels=targets).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        metrics = classification_metrics(
            validation_truth,
            model.probabilities(validation["text"].tolist(), config.batch_size),
            model.labels,
        )
        # Keep aggregate metrics only; never log support text or the full confusion matrix.
        row = {
            "epoch": epoch,
            "training_loss": total_loss / len(train),
            "validation_macro_f1": metrics["macro_f1"],
            "validation_accuracy": metrics["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch, stale = metrics["macro_f1"], epoch, 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.model.load_state_dict(best_state)
    return {
        "configuration": asdict(config),
        "random_seed": config.seed,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "selection_metric": "validation_macro_f1",
        "history": history,
        "training_seconds": perf_counter() - start,
        "training_time_scope": (
            "optimizer steps, validation and best-state restoration; excludes download/build/save"
        ),
        "smoke": config.smoke,
        "runtime": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("torch", "transformers", "numpy", "tokenizers", "safetensors")
            },
        },
    }
