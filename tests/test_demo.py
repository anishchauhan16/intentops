from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intentops.api import create_app
from intentops.config import Settings
from intentops.http import MAX_BODY_BYTES
from intentops.router import Router


def test_demo_examples_assets_and_security_headers(router: Router) -> None:
    with TestClient(create_app(router=router)) as client:
        page = client.get("/?example=pin")
        assert page.status_code == 200
        assert "How do I change my PIN?" in page.text
        assert 'action="/demo"' in page.text
        assert page.headers["cache-control"] == "no-store"
        assert "form-action 'self'" in page.headers["content-security-policy"]
        assert "default-src 'none'" in page.headers["content-security-policy"]
        assert client.get("/static/demo.css").status_code == 200
        # User-provided example query strings are never reflected in the page.
        assert "ATTACK" not in client.get("/?example=ATTACK").text
        assert client.get("/docs").status_code == 200


def test_demo_prediction_and_escaped_support_text(router: Router) -> None:
    with TestClient(create_app(router=router)) as client:
        page = client.post("/demo", data={"text": "reset my pin code"})
        assert page.status_code == 200
        assert "change pin" in page.text
        assert "Classifier confidence" in page.text
        assert "Classifier alternatives" in page.text
        attack = '</textarea><script>alert("x")</script>'
        response = client.post("/demo", data={"text": attack})
        assert response.status_code == 200
        assert "<script>" not in response.text
        assert "&lt;script&gt;" in response.text
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "body", ["", "text=+", "text=" + "x" * 2001, "other=test", "text=a&text=b"]
)
def test_demo_invalid_inputs(router: Router, body: str) -> None:
    with TestClient(create_app(router=router)) as client:
        response = client.post(
            "/demo", content=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == 422
        assert "Enter 1–2,000 nonblank characters." in response.text


def test_demo_missing_model_and_wrong_media_type(tmp_path: Path, router: Router) -> None:
    with TestClient(create_app(Settings(artifact_dir=tmp_path))) as client:
        assert "Model unavailable" in client.get("/").text
        assert client.post("/demo", data={"text": "hello"}).status_code == 503
    with TestClient(create_app(router=router)) as client:
        assert client.post("/demo", json={"text": "hello"}).status_code == 415


def test_body_limit_includes_streamed_requests(router: Router) -> None:
    with TestClient(create_app(router=router)) as client:
        response = client.post("/predict", content=b"x" * (MAX_BODY_BYTES + 1))
        assert response.status_code == 413
        assert response.json()["detail"] == "Request body exceeds 1 MiB."
        response = client.post("/demo", content=iter([b"x" * MAX_BODY_BYTES, b"x"]))
        assert response.status_code == 413
        # A valid maximum-sized Unicode batch also fits the transport limit.
        assert client.post("/batch-predict", json={"texts": ["😀" * 2000] * 32}).status_code == 200
