import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from intentops.api import create_app
from intentops.config import Settings
from intentops.model import Baseline
from intentops.router import Router
from intentops.schemas import MAX_BATCH_SIZE
from intentops.semantic import SemanticIndex


def test_health_metadata_and_prediction(router: Router) -> None:
    with TestClient(create_app(router=router)) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "model_loaded": True,
            "detail": None,
        }
        metadata = client.get("/metadata").json()
        assert metadata["class_count"] == 3
        assert metadata["semantic_available"] is False
        response = client.post("/predict", json={"text": "reset my pin code"})
        assert response.status_code == 200
        prediction = response.json()
        assert set(prediction) == {
            "intent",
            "confidence",
            "needs_review",
            "top_k",
            "model_version",
            "latency_ms",
            "source",
        }
        assert prediction["intent"] == "change_pin"
        assert 0 <= prediction["confidence"] <= 1
        assert prediction["model_version"] == metadata["model_version"]
        assert set(prediction["top_k"][0]) == {"intent", "probability"}
        assert prediction["latency_ms"] >= 0


@pytest.mark.parametrize(
    "body",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "x" * 2001},
        {"text": 123},
        {},
        {"text": None},
        {"text": "hello", "extra": True},
    ],
)
def test_predict_validation(router: Router, body: dict) -> None:
    with TestClient(create_app(router=router)) as client:
        assert client.post("/predict", json=body).status_code == 422


def test_batch_bounds_and_predictions(router: Router) -> None:
    with TestClient(create_app(router=router)) as client:
        result = client.post("/batch-predict", json={"texts": ["reset pin", "where is my card"]})
        assert result.status_code == 200
        assert len(result.json()["predictions"]) == 2
        for texts in ([], ["hello"] * (MAX_BATCH_SIZE + 1), ["valid", "  "]):
            assert client.post("/batch-predict", json={"texts": texts}).status_code == 422
        assert (
            client.post("/batch-predict", json={"texts": ["pin"] * MAX_BATCH_SIZE}).status_code
            == 200
        )


def test_missing_artifacts_unavailable(tmp_path: Path) -> None:
    with TestClient(create_app(Settings(artifact_dir=tmp_path))) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False
        assert "train.py" in response.json()["detail"]
        assert client.get("/metadata").status_code == 503
        assert client.post("/predict", json={"text": "hello"}).status_code == 503


def test_lifespan_loads_real_artifact(artifact_dir: Path) -> None:
    with TestClient(create_app(Settings(artifact_dir=artifact_dir))) as client:
        assert client.get("/health").status_code == 200


def test_enabled_semantic_missing_fails_health(artifact_dir: Path) -> None:
    with TestClient(
        create_app(Settings(artifact_dir=artifact_dir, semantic_enabled=True))
    ) as client:
        assert client.get("/health").status_code == 503


def test_semantic_api_loads_only_local_mock_encoder(
    artifact_dir: Path,
    baseline: Baseline,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_artifacts = tmp_path / "artifacts"
    shutil.copytree(artifact_dir, local_artifacts)
    SemanticIndex(
        np.asarray([[1.0, 0]]), ["public example"], ["change_pin"], "mock", baseline.version
    ).save(local_artifacts / "semantic.npz")
    encoder = Mock()
    encoder.encode.return_value = np.asarray([[1.0, 0]])
    monkeypatch.setattr(SemanticIndex, "ensure_encoder", lambda self: encoder)
    settings = Settings(artifact_dir=local_artifacts, semantic_enabled=True, review_threshold=1)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metadata").json()["semantic_available"] is True
        response = client.post("/predict", json={"text": "reset pin"})
        assert response.status_code == 200
        body = response.json()
        assert body["source"] == "semantic"
        assert body["needs_review"] is False
        assert body["semantic_matches"][0]["text"] == "public example"
