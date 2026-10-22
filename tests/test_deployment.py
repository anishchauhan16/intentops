import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from intentops import serve
from intentops.model import Baseline
from scripts.package_demo import package_demo


def test_host_port_single_worker_and_access_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    launch = Mock()
    monkeypatch.setattr(serve.uvicorn, "run", launch)
    monkeypatch.setenv("PORT", "10000")
    serve.main()
    assert launch.call_args.kwargs["port"] == 10000
    assert launch.call_args.kwargs["workers"] == 1
    assert launch.call_args.kwargs["access_log"] is False
    assert launch.call_args.kwargs["limit_concurrency"] == 16
    monkeypatch.setenv("PORT", "0")
    with pytest.raises(ValueError, match="PORT"):
        serve.main()


def test_sklearn_runtime_mismatch_rejected(
    artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("intentops.model.version", lambda name: "wrong-version")
    with pytest.raises(ValueError, match="training runtime"):
        Baseline.load(artifact_dir)


def test_release_excludes_local_secrets_raw_data_and_caches(
    tmp_path: Path,
    artifact_dir: Path,
) -> None:
    import shutil

    root = tmp_path / "source"
    root.mkdir()
    names = [
        "README.md",
        "AGENTS.md",
        "LICENSE",
        "NOTICE.md",
        "pyproject.toml",
        "requirements-runtime.txt",
        "Makefile",
        "Dockerfile",
        ".dockerignore",
        ".env.example",
        "render.yaml",
        ".github/workflows/ci.yml",
        "reports/README.md",
        "reports/summary.json",
    ]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture")
    shutil.copytree(artifact_dir, root / "artifacts")
    (root / ".env").write_text("PRIVATE_TOKEN=synthetic-secret")
    (root / "artifacts/semantic.npz").write_bytes(b"large optional artifact")
    output = tmp_path / "release"
    package_demo(root, output)
    assert (output / "artifacts/baseline.joblib").exists()
    assert not (output / ".env").exists()
    assert not (output / "artifacts/semantic.npz").exists()
    assert not (output / ".venv").exists()
    assert (
        json.loads((output / "release.json").read_text())["deployment_status"]
        == "prepared_locally_only"
    )
    with pytest.raises(ValueError, match="already exists"):
        package_demo(root, output)
