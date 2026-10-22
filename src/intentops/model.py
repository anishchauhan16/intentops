"""One pipeline for training, persistence, and inference."""

import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intentops.config import TrainingConfig
from intentops.data import sha256_file, split_development
from intentops.metrics import classification_metrics
from intentops.schemas import IntentScore


def build_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.ngram_range,
                    sublinear_tf=config.sublinear_tf,
                    min_df=config.min_df,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config.c, max_iter=config.max_iter, random_state=config.seed, solver="lbfgs"
                ),
            ),
        ]
    )


class Baseline:
    def __init__(self, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.labels: list[str] = [str(label) for label in pipeline.classes_]
        self.version = str(metadata["model_version"])

    def probabilities(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(texts))

    def top_k(self, text: str, k: int = 3) -> list[IntentScore]:
        if k < 1:
            raise ValueError("k must be positive.")
        probabilities = self.probabilities([text])[0]
        indices = np.argsort(-probabilities, kind="stable")[:k]
        return [
            IntentScore(intent=self.labels[i], probability=float(probabilities[i])) for i in indices
        ]

    @classmethod
    def load(cls, artifact_dir: Path) -> "Baseline":
        model_path = artifact_dir / "baseline.joblib"
        metadata_path = artifact_dir / "metadata.json"
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing baseline.joblib or metadata.json in {artifact_dir}. "
                "Run python scripts/train.py first."
            )
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("model_sha256") != sha256_file(model_path):
            raise ValueError("Model checksum differs from metadata; retrain the artifact bundle.")
        trained_version = metadata.get("runtime", {}).get("scikit-learn")
        if trained_version != version("scikit-learn"):
            raise ValueError(
                "scikit-learn differs from the training runtime. Install requirements-runtime.txt "
                "matching the artifacts, or retrain and evaluate with the new runtime."
            )
        pipeline = joblib.load(model_path)  # Only load artifacts from a trusted source.
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Baseline artifact must be an sklearn Pipeline.")
        baseline = cls(pipeline, metadata)
        if metadata.get("label_names") != baseline.labels:
            raise ValueError("Model labels differ from metadata; retrain the artifact bundle.")
        return baseline


def train_baseline(
    development: pd.DataFrame, artifact_dir: Path, config: TrainingConfig
) -> tuple[Baseline, pd.DataFrame]:
    train, validation = split_development(development, config.validation_size, config.seed)
    pipeline = build_pipeline(config)
    pipeline.fit(train["text"], train["intent"])
    labels = [str(label) for label in pipeline.classes_]
    metrics = classification_metrics(
        validation["intent"].tolist(), pipeline.predict_proba(validation["text"]), labels
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "baseline.joblib"
    joblib.dump(pipeline, model_path)
    checksum = sha256_file(model_path)
    metadata = {
        "model_name": "tfidf-logistic-regression",
        "model_version": f"baseline-{checksum[:12]}",
        "model_sha256": checksum,
        "algorithm": "TfidfVectorizer + LogisticRegression",
        "configuration": asdict(config),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "class_count": len(labels),
        "label_names": labels,
        "random_seed": config.seed,
        "validation_metrics": metrics,
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("scikit-learn", "pandas", "numpy", "joblib")},
        },
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return Baseline(pipeline, metadata), train
