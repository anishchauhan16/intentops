import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from intentops.config import TrainingConfig
from intentops.model import Baseline, train_baseline


def test_tiny_model_trains_and_predicts(tmp_path: Path, development: pd.DataFrame) -> None:
    model, train = train_baseline(development, tmp_path, TrainingConfig(min_df=1))
    assert len(train) == 24
    assert model.top_k("reset my pin code")[0].intent == "change_pin"
    assert model.metadata["validation_rows"] == 6
    assert model.metadata["class_count"] == 3
    assert model.metadata["random_seed"] == 42
    assert 0 <= model.metadata["validation_metrics"]["accuracy"] <= 1
    restored = Baseline.load(tmp_path)
    np.testing.assert_allclose(
        model.probabilities(["where is my card"]), restored.probabilities(["where is my card"])
    )
    assert model.version == restored.version


def test_top_k_sorted_and_bounded(baseline: Baseline) -> None:
    predictions = baseline.top_k("when does my transfer arrive", 20)
    scores = [item.probability for item in predictions]
    assert len(predictions) == 3
    assert scores == sorted(scores, reverse=True)
    assert sum(scores) == pytest.approx(1)
    with pytest.raises(ValueError, match="positive"):
        baseline.top_k("test", 0)


def test_missing_artifacts_helpful(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train.py"):
        Baseline.load(tmp_path)


def test_corrupt_model_and_labels_rejected(tmp_path: Path, development: pd.DataFrame) -> None:
    train_baseline(development, tmp_path, TrainingConfig(min_df=1))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["label_names"] = ["wrong"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="labels differ"):
        Baseline.load(tmp_path)
    (tmp_path / "baseline.joblib").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        Baseline.load(tmp_path)
