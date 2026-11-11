"""Shared CPU timing, evaluation exports and transparent comparison reports."""

import hashlib
import json
import platform
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from intentops.metrics import classification_metrics, confusion_pairs, selective_accuracy


def cpu_timing(
    predict: Callable[[list[str]], np.ndarray],
    texts: list[str],
    batch_sizes: tuple[int, ...] = (1, 16),
    warmup: int = 3,
    repeats: int = 3,
) -> dict[str, Any]:
    """Time identical text batches including preprocessing, excluding startup and warmup."""
    if not texts or min(batch_sizes, default=0) < 1 or warmup < 1 or repeats < 1:
        raise ValueError("Timing requires texts, positive batch sizes, warmup and repeats.")
    results = []
    with threadpool_limits(limits=1):
        for size in batch_sizes:
            if len(texts) % size:
                raise ValueError("Use an example count divisible by every batch size.")
            batches = [texts[start : start + size] for start in range(0, len(texts), size)]
            for index in range(warmup):
                predict(batches[index % len(batches)])
            samples = []
            for _ in range(repeats):
                for batch in batches:
                    start = perf_counter()
                    predict(batch)
                    samples.append((perf_counter() - start) * 1000 / len(batch))
            results.append(
                {
                    "batch_size": size,
                    "mean_ms_per_example": float(np.mean(samples)),
                    "p95_ms_per_example": float(np.percentile(samples, 95)),
                    "measured_batches": len(samples),
                }
            )
    return {
        "protocol": "sequential_cpu_preprocessing_forward_probabilities_v1",
        "scope": (
            "in-process CPU; preprocessing and probabilities included; "
            "startup/warmup excluded; not network SLA"
        ),
        "p95_definition": (
            "95th percentile of batch elapsed milliseconds divided by batch size; "
            "amortized for batches >1"
        ),
        "cpu_threads": 1,
        "platform": platform.platform(),
        "examples": len(texts),
        "examples_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode()
        ).hexdigest(),
        "warmup_batches_per_size": warmup,
        "repeats": repeats,
        "batches": results,
    }


def evaluate_probabilities(
    frame: pd.DataFrame, probabilities: np.ndarray, labels: list[str], output: Path
) -> dict[str, Any]:
    truth = frame["intent"].tolist()
    metrics = classification_metrics(truth, probabilities, labels)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = np.asarray(labels)[order[:, 0]].tolist()
    confidence = probabilities.max(axis=1).tolist()
    gate = selective_accuracy(truth, predicted, confidence, thresholds=(0.7,)).iloc[0]
    output.mkdir(parents=True, exist_ok=True)
    confusion_pairs(truth, predicted).to_csv(output / "confusion_pairs.csv", index=False)
    errors = [
        {
            "text": text,
            "true_intent": actual,
            "predicted_intent": prediction,
            "confidence": score,
            "top_alternatives": json.dumps(
                [
                    {"intent": labels[index], "probability": float(probabilities[row, index])}
                    for index in order[row, :3]
                ]
            ),
        }
        for row, (text, actual, prediction, score) in enumerate(
            zip(frame["text"], truth, predicted, confidence, strict=True)
        )
        if actual != prediction
    ]
    pd.DataFrame(
        errors,
        columns=["text", "true_intent", "predicted_intent", "confidence", "top_alternatives"],
    ).to_csv(output / "errors.csv", index=False)
    metrics["gate"] = {
        "threshold": 0.7,
        "coverage": float(gate["coverage"]),
        "accepted_accuracy": None
        if pd.isna(gate["accepted_accuracy"])
        else float(gate["accepted_accuracy"]),
        "number_accepted": int(gate["number_accepted"]),
        "number_review": int(gate["number_review"]),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def write_comparison(report: dict[str, Any], output: Path) -> None:
    """Smoke observations are recorded separately and never promoted to benchmark results."""
    pending = report["smoke"]
    columns: dict[str, dict[str, Any]] = {}
    for name in ("baseline", "transformer"):
        measured = report[name]
        columns[name] = {
            "accuracy": None if pending else measured["metrics"]["accuracy"],
            "macro_f1": None if pending else measured["metrics"]["macro_f1"],
            "weighted_f1": None if pending else measured["metrics"]["weighted_f1"],
            "top_3_accuracy": None if pending else measured["metrics"]["top_3_accuracy"],
            "coverage_at_0_70": None if pending else measured["metrics"]["gate"]["coverage"],
            "accepted_accuracy_at_0_70": None
            if pending
            else measured["metrics"]["gate"]["accepted_accuracy"],
            "latency": None if pending else measured["latency"],
            "artifact_bytes": measured["artifact_bytes"]
            if name == "baseline" or not pending
            else None,
            "training_seconds": None
            if name == "baseline" or pending
            else measured["training_seconds"],
            "training_hardware": None
            if name == "baseline" or pending
            else measured["training_hardware"],
        }
    document = {
        "status": "PENDING_FULL_TRAINING_AND_FINAL_TEST"
        if pending
        else f"MEASURED_{report['split'].upper()}_ONLY",
        "split": None if pending else report["split"],
        "pending_reason": (
            "Only a reduced training/validation smoke run was performed; "
            "official test was not opened."
        )
        if pending
        else None,
        "baseline_training_note": (
            "PENDING: original duration/hardware not recorded; frozen baseline was not retrained."
        ),
        "artifact_size_scope": (
            "baseline.joblib + metadata; transformer checkpoint + tokenizer + metadata; "
            "no dataset/cache/receipt"
        ),
        "comparison": columns,
        "run_evidence": report,
    }
    rows = [
        "# Transformer versus frozen baseline",
        "",
        f"Status: **{document['status']}**.",
        "",
        "All null fields mean **PENDING**, not zero.",
        (
            "Official test results from the single frozen final run."
            if report["split"] == "test" and not pending
            else "Official test results are pending until the single frozen final run."
        ),
        "",
        "| Metric | Baseline | Transformer |",
        "| --- | ---: | ---: |",
    ]

    def display(value: Any) -> str:
        return (
            "PENDING"
            if value is None
            else f"{value:.6f}"
            if isinstance(value, float)
            else str(value)
        )

    for key in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top_3_accuracy",
        "coverage_at_0_70",
        "accepted_accuracy_at_0_70",
        "artifact_bytes",
        "training_seconds",
    ):
        rows.append(
            f"| {key} | {display(columns['baseline'][key])} "
            f"| {display(columns['transformer'][key])} |"
        )
    for size in report["timing_configuration"]["batch_sizes"]:
        for metric in ("mean_ms_per_example", "p95_ms_per_example"):
            values = []
            for name in ("baseline", "transformer"):
                latency = columns[name]["latency"]
                values.append(
                    display(
                        next(row[metric] for row in latency["batches"] if row["batch_size"] == size)
                    )
                    if latency
                    else "PENDING"
                )
            rows.append(f"| CPU batch {size} {metric} | {values[0]} | {values[1]} |")
    hardware = columns["transformer"]["training_hardware"]
    hardware_text = (
        f"{hardware.get('device')}, {hardware.get('machine')}, "
        f"{hardware.get('torch_cpu_threads')} CPU threads"
        if hardware
        else "PENDING"
    )
    rows.append(f"| Training hardware | PENDING | {hardware_text} |")
    rows.extend(
        [
            "",
            "Baseline training duration/hardware: **PENDING**; no baseline retraining.",
            "",
            "Latency uses the same CPU, examples, threads, batches and warmups for both models.",
            "Batch p95 is amortized batch time per example, not per-request network tail latency.",
            "",
            "See `run_evidence` in the JSON for actual evaluation observations, hardware, "
            "training time and shared CPU timings.",
            "Smoke observations are pipeline checks, not transformer benchmark results.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(document, indent=2) + "\n")
    output.with_suffix(".md").write_text("\n".join(rows) + "\n")
