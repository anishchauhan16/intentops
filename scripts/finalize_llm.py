"""Frozen Phase 2 full validation, predeclared tau selection and one-shot test."""

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from intentops.data import sha256_file
from intentops.llm_cache import ResponseCache, atomic_json
from intentops.llm_evaluation import execute, quality
from intentops.llm_final import (
    CacheOnly,
    claim_test,
    create_freeze,
    prompt_contract,
    read_final_test,
    save_comparison,
    select_tau,
    source_identity,
    summarize_run,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["validation", "test"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("../data/raw"))
    args = parser.parse_args()
    start = perf_counter()
    seed_everything(threads=2)
    code = source_identity()
    config = load_config(Path("configs/llm-pilot.json"))
    pilot = json.loads(Path("reports/phase2_pilot.json").read_text())
    if asdict(config) != pilot["configuration"]:
        raise ValueError(
            "Pilot configuration changed; no substitution or prompt retuning permitted."
        )
    for name, expected in pilot["runtime"].items():
        if version(name) != expected:
            raise ValueError(f"Runtime version changed for {name}; stop before evaluation.")
    baseline = Baseline.load(Path("artifacts"))
    train, validation, labels, provenance = development_data(args.data_dir, Path("artifacts"))
    original = subprocess.check_output(
        ["git", "show", "ade84f9:src/intentops/llm_path.py"], text=True
    )
    if prompt_contract(config, labels, original) != prompt_contract(config, labels):
        raise ValueError("Prompt/schema differs from the pilot source.")
    binding = index_binding(
        config,
        labels,
        provenance,
        Path("configs/intent-descriptions-v1.json"),
        train.index.tolist(),
    )
    if binding != pilot["index_binding"]:
        raise ValueError("Index/data binding changed since the pilot.")
    metadata_path = Path("artifacts/transformer/metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata["model_version"] != pilot["transformer_version"]
        or baseline.version != pilot["baseline_version"]
        or metadata["smoke"]
        or metadata["baseline_sha256"] != provenance["baseline_sha256"]
        or metadata["manifest_sha256"] != provenance["manifest_sha256"]
    ):
        raise ValueError("Frozen baseline/transformer identity differs.")
    transformer = TransformerModel.load(Path("artifacts/transformer"), labels)
    models = {
        "baseline_version": baseline.version,
        "baseline_sha256": provenance["baseline_sha256"],
        "transformer_version": metadata["model_version"],
        "transformer_metadata_sha256": sha256_file(metadata_path),
        "transformer_sha256": metadata["model_sha256"],
    }
    retriever = ChromaRetriever(
        Path("artifacts/llm/index"), LocalEncoder(config, Path(".hf-cache")), binding
    )
    provider = OllamaProvider(config)
    provider.verify()
    if provider.version != pilot["hardware"]["ollama_version"]:
        raise ValueError("Ollama version differs from the verified pilot runtime.")
    cache = ResponseCache(Path("artifacts/llm/responses"))
    freeze_path = Path("reports/llm_freeze.json")
    validation_path = Path("reports/llm_full_validation.json")
    attempt_path = Path("artifacts/llm/final_test_receipt.json")
    if args.stage == "validation":
        if validation_path.exists() or freeze_path.exists():
            raise ValueError("Full validation or freeze already exists; do not overwrite evidence.")
        pilot_frame = validation.loc[pilot["row_ids"]]
        graph = build_graph(
            config, labels, baseline, retriever, provider, CacheOnly(cache.directory)
        )
        cached, cached_seconds = execute(
            graph, pilot_frame, Path("reports/llm-runs/full-validation/pilot-cache.json")
        )
        if (
            not all(row["cache_hit"] for row in cached)
            or [row["row_sha256"] for row in cached] != pilot["row_hashes"]
        ):
            raise ValueError("Pilot cache/sample identity mismatch; remaining evaluation stopped.")
        remaining = validation.drop(index=pilot["row_ids"])
        if len(remaining) != 1751:
            raise ValueError("Expected exactly 1,751 remaining validation rows.")
        print(
            json.dumps(
                {
                    "stage": "validation",
                    "verified_pilot_cache_hits": len(cached),
                    "remaining": len(remaining),
                }
            ),
            flush=True,
        )
        graph = build_graph(config, labels, baseline, retriever, provider, cache)
        live, live_seconds = execute(
            graph, remaining, Path("reports/llm-runs/full-validation/remaining.json")
        )
        if any(row["cache_hit"] for row in live):
            raise ValueError("Unexpected existing remaining-row cache entries; inspect provenance.")
        by_id = {row["row_id"]: row for row in [*cached, *live]}
        records = [by_id[int(identity)] for identity in validation.index]
        selection = select_tau(records, validation.intent.tolist())
        frame = validation
        elapsed = cached_seconds + live_seconds
        tau = selection["chosen"]["tau"]
    else:
        frozen = claim_test(
            attempt_path, freeze_path, config, labels, binding, validation_path, models
        )
        config = replace(config, tau=frozen["configuration"]["tau"])
        frame = read_final_test(
            args.data_dir / "test.parquet", provenance["dataset_checksums"]["test"], labels
        )
        graph = build_graph(config, labels, baseline, retriever, provider, cache)
        records, elapsed = execute(graph, frame, Path("reports/llm-runs/final-test/compute.json"))
        tau = config.tau
        if tau is None:
            raise ValueError("Final test requires the frozen selected tau.")
    # The graph already evaluated the frozen baseline on each exact example.
    comparisons: dict[str, Any] = {
        "baseline": quality(
            frame.intent.tolist(), [row["baseline_label"] for row in records], labels
        )
    }
    predictions = np.asarray(labels)[
        transformer.probabilities(frame.text.tolist()).argmax(axis=1)
    ].tolist()
    comparisons["transformer"] = quality(frame.intent.tolist(), predictions, labels)
    comparisons["llm_rag"] = quality(
        frame.intent.tolist(), [row["intent"] for row in records], labels
    )
    report = summarize_run(args.stage, frame, records, elapsed, comparisons, tau)
    report["overall_wall_seconds"] = perf_counter() - start
    report["created_at"] = datetime.now(UTC).isoformat()
    report["code"] = code
    report["hardware"] = {
        "platform": platform.platform(),
        "processor": "Apple M5",
        "ram_gib": 16,
        "embedding_device": "cpu",
        "transformer_device": "cpu",
        "ollama_version": provider.version,
    }
    report["runtime"] = pilot["runtime"]
    if args.stage == "validation":
        report["selection"] = selection
        atomic_json(validation_path, report)
        create_freeze(freeze_path, config, labels, binding, selection, validation_path, models)
        print(
            json.dumps(
                {
                    "stage": "validation_completed_and_frozen",
                    "comparison": comparisons,
                    "selection": {k: v for k, v in selection.items() if k != "sweep"},
                }
            ),
            flush=True,
        )
    else:
        published = json.loads(Path("reports/transformer_vs_baseline.json").read_text())
        report["reproduction"] = {
            name: {
                "expected_accuracy": published["comparison"][name]["accuracy"],
                "observed_accuracy": comparisons[name]["accuracy"],
                "matches": abs(
                    comparisons[name]["accuracy"] - published["comparison"][name]["accuracy"]
                )
                < 1e-12,
            }
            for name in ("baseline", "transformer")
        }
        validation_report = json.loads(validation_path.read_text())
        document = {
            "status": "SINGLE_FINAL_TEST_COMPLETED",
            "api_cost_usd": 0.0,
            "freeze_receipt": json.loads(freeze_path.read_text()),
            "validation": validation_report,
            "test": report,
            "selection": validation_report["selection"],
            "limitations": [
                "One frozen local experiment; no test-based model/prompt/threshold selection.",
                "Validation-selected accuracy and Wilson intervals are exploratory.",
                "Local latency is not comparable to hosted APIs or Phase 1 CPU timing.",
            ],
        }
        save_comparison(Path("reports/llm_vs_baseline_transformer"), document)
        atomic_json(
            attempt_path,
            {
                **json.loads(attempt_path.read_text()),
                "status": "completed",
                "completed_at": datetime.now(UTC).isoformat(),
                "freeze_sha256": sha256_file(freeze_path),
                "comparison_sha256": sha256_file(Path("reports/llm_vs_baseline_transformer.json")),
                "rows": len(frame),
                "code_commit": code["code_commit"],
            },
        )
        print(
            json.dumps(
                {
                    "stage": "single_final_test_completed",
                    "comparison": comparisons,
                    "gate": report["gate"],
                    "reproduction": report["reproduction"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
