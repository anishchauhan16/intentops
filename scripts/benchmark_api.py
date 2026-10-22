"""Measure warmed, sequential, in-process HTTP handling (no network transport)."""

import argparse
import json
import logging
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import numpy as np
from fastapi.testclient import TestClient

from intentops.api import create_app
from intentops.config import Settings

EXAMPLES = (
    "I am still waiting for my card",
    "How do I change my PIN?",
    "My transfer has not arrived",
    "I lost my phone",
    "The cash machine kept my card",
    "I do not recognize this payment",
)


def benchmark(requests: int, warmup: int, artifact_dir: Path) -> dict[str, Any]:
    if requests < 1 or warmup < 1:
        raise ValueError("Request count and warmup must be positive.")
    app = create_app(Settings(artifact_dir=artifact_dir))
    with TestClient(app) as client:
        client.get("/health").raise_for_status()
        metadata = client.get("/metadata").json()
        samples = []
        for index in range(warmup + requests):
            start = perf_counter()
            response = client.post("/predict", json={"text": EXAMPLES[index % len(EXAMPLES)]})
            response.raise_for_status()
            elapsed = (perf_counter() - start) * 1000
            if index >= warmup:
                samples.append(elapsed)
    return {
        "measurement": "in_process_fastapi_testclient_round_trip",
        "includes": "validation, routing, logging, serialization, TestClient thread bridge",
        "excludes": "network transport, model startup, concurrency and load testing",
        "mode": "baseline",
        "model_version": metadata["model_version"],
        "request_count": requests,
        "warmup_count": warmup,
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        report = benchmark(args.requests, args.warmup, args.artifact_dir)
    except (ValueError, httpx.HTTPStatusError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}. Train the baseline and check /health.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
