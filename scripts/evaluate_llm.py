"""Validation-only smoke/pilot evaluation; official test access is unavailable."""

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from intentops.llm_cache import ResponseCache, fingerprint
from intentops.llm_evaluation import (
    execute,
    extrapolation,
    gate_diagnostics,
    quality,
    stratified_sample,
    summarize_compute,
    write_report,
)
from intentops.llm_path import (
    ChromaRetriever,
    LocalEncoder,
    OllamaProvider,
    build_graph,
    index_binding,
    load_config,
)
from intentops.model import Baseline
from intentops.transformer_model import TransformerModel, development_data, seed_everything


def require_smoke(path: Path, configuration: dict[str, Any], binding: dict[str, Any]) -> None:
    smoke = json.loads(path.read_text())
    timing = smoke["compute"]["live"]["latency"]["end_to_end"]
    if (
        not smoke["smoke_gate_passed"]
        or smoke["rows"] != 20
        or timing["examples"] != 20
        or timing["mean_seconds"] > 10
        or smoke["configuration"] != configuration
        or smoke["index_binding"] != binding
    ):
        raise ValueError(
            "A passing 20-live-row smoke gate for this exact configuration is required."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="pilot")
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--smoke-report", type=Path, default=Path("reports/phase2_smoke.json"))
    parser.add_argument("--transformer-dir", type=Path, default=Path("artifacts/transformer"))
    parser.add_argument("--data-dir", type=Path, default=Path("../data/raw"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/llm-pilot.json"))
    parser.add_argument(
        "--descriptions", type=Path, default=Path("configs/intent-descriptions-v1.json")
    )
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/llm/index"))
    parser.add_argument("--embedding-cache", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/llm/responses"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.run_dir = args.run_dir or Path("reports/llm-runs") / args.mode
    args.report = args.report or Path(f"reports/phase2_{args.mode}")
    if (args.run_dir / "compute.json").exists() or args.report.with_suffix(".json").exists():
        raise ValueError("Run/report already exists; use new --run-dir and --report paths.")
    overall_start = perf_counter()
    seed_everything(threads=2)
    config = load_config(args.config)
    baseline = Baseline.load(args.baseline_dir)
    train, validation, labels, provenance = development_data(args.data_dir, args.baseline_dir)
    binding = index_binding(config, labels, provenance, args.descriptions, train.index.tolist())
    if args.mode == "pilot":
        require_smoke(args.smoke_report, asdict(config), binding)
    frame = stratified_sample(validation, 20 if args.mode == "smoke" else 250)
    comparison = {}
    transformer_record = None
    if args.mode == "pilot":
        transformer_record = json.loads((args.transformer_dir / "metadata.json").read_text())
        if (
            transformer_record["smoke"]
            or transformer_record["baseline_sha256"] != provenance["baseline_sha256"]
            or transformer_record["manifest_sha256"] != provenance["manifest_sha256"]
        ):
            raise ValueError(
                "Expected the full frozen transformer bound to the current baseline/data."
            )
        transformer = TransformerModel.load(args.transformer_dir, labels)
        for name, predict in (
            ("baseline", baseline.probabilities),
            ("transformer", transformer.probabilities),
        ):
            predictions = np.asarray(labels)[predict(frame.text.tolist()).argmax(axis=1)].tolist()
            comparison[name] = quality(frame.intent.tolist(), predictions, labels)
    encoder = LocalEncoder(config, args.embedding_cache)
    retriever = ChromaRetriever(args.index_dir, encoder, binding)
    provider = OllamaProvider(config)
    provider.verify()
    graph = build_graph(
        config, labels, baseline, retriever, provider, ResponseCache(args.cache_dir)
    )
    records, elapsed = execute(graph, frame, args.run_dir / "compute.json")
    compute = summarize_compute(records, elapsed)
    timing = compute["live"]["latency"]["end_to_end"]
    passed = timing["examples"] == 20 and timing["mean_seconds"] <= 10
    report: dict[str, Any] = {
        "status": "MEASURED_VALIDATION_PILOT_TAU_PENDING"
        if args.mode == "pilot"
        else "SMOKE_GATE_PASSED"
        if passed
        else "STOP_SMOKE_GATE_FAILED",
        "split": "validation",
        "rows": len(frame),
        "represented_classes": frame.intent.nunique(),
        "sampling": "proportional quotas, seed 42; <77 rows cannot cover 77 labels",
        "configuration": asdict(config),
        "configuration_sha256": fingerprint(asdict(config)),
        "index_binding": binding,
        "provenance": provenance,
        "row_ids": [row["row_id"] for row in records],
        "row_hashes": [row["row_sha256"] for row in records],
        "sample_sha256": fingerprint([row["row_sha256"] for row in records]),
        "invalid_output_count": sum(row["intent"] is None for row in records),
        "failure_count": sum(bool(row["failure"]) for row in records),
        "compute": compute,
        "hardware": {
            "platform": platform.platform(),
            "processor": "Apple M5",
            "ram_gb": 16,
            "embedding_device": "cpu",
            "ollama_version": provider.version,
        },
        "runtime": {
            name: version(name)
            for name in (
                "langgraph",
                "chromadb",
                "sentence-transformers",
                "torch",
                "transformers",
                "httpx",
            )
        },
        "overall_wall_seconds": perf_counter() - overall_start,
        "wall_scope": "overall includes setup/comparisons; compute wall covers graph loop",
    }
    if args.mode == "smoke":
        report["smoke_gate_passed"] = passed
        report["note"] = "No smoke accuracy/F1 is computed or reported; latency/format gate only."
    else:
        comparison["llm_rag"] = quality(
            frame.intent.tolist(), [r["intent"] for r in records], labels
        )
        report.update(
            {
                "comparison": comparison,
                "gate": gate_diagnostics(records, frame.intent.tolist()),
                "time_extrapolation": extrapolation(compute),
                "baseline_version": baseline.version,
                "transformer_version": transformer_record["model_version"]
                if transformer_record
                else None,
                "limitations": [
                    "250 validation rows only; no tau selection or full validation.",
                    "Agreement-only accuracy is diagnostic; no tuned acceptance threshold.",
                    "99% is a target; inspect support and Wilson interval, not assurance.",
                    "Mean/p95 extrapolation is a planning range, not an interval on total runtime.",
                    "Local timings differ from hosted APIs and the Phase 1 CPU protocol.",
                    "Models were loaded, never retrained. Official test split was never opened.",
                ],
            }
        )
        residency = provider.client.get("/api/ps")
        residency.raise_for_status()
        report["hardware"]["ollama_residency"] = residency.json()["models"]
    write_report(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "latency": timing,
                "invalid_output_count": report["invalid_output_count"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
