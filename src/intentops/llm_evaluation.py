"""Validation-only local experiment sampling, compute accounting and diagnostics."""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from intentops.llm_cache import atomic_json, fingerprint
from intentops.llm_path import invoke


def stratified_sample(frame: pd.DataFrame, count: int, seed: int = 42) -> pd.DataFrame:
    """Proportional largest-remainder quotas; <77 rows necessarily omit some classes."""
    if count < 1 or count > len(frame) or not frame.index.is_unique:
        raise ValueError("Sample size/index invalid.")
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("intent", sort=True))
    exact = np.asarray([len(group) * count / len(frame) for _, group in groups])
    quotas = np.floor(exact).astype(int)
    tie = rng.random(len(groups))
    order = sorted(range(len(groups)), key=lambda i: (-(exact[i] - quotas[i]), tie[i]))
    for i in order[: count - int(quotas.sum())]:
        quotas[i] += 1
    selected = pd.concat(
        [
            group.sample(n=int(quota), random_state=seed)
            for (_, group), quota in zip(groups, quotas, strict=True)
            if quota
        ]
    )
    return selected.sample(frac=1, random_state=seed).copy()


def latency(values: list[float]) -> dict[str, Any]:
    return {
        "examples": len(values),
        "mean_seconds": float(np.mean(values)) if values else None,
        "p95_seconds": float(np.percentile(values, 95)) if values else None,
    }


def wilson(correct: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    fraction = correct / count
    denominator = 1 + z * z / count
    centre = (fraction + z * z / (2 * count)) / denominator
    half = z * np.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count))
    half /= denominator
    return [max(0.0, centre - float(half)), min(1.0, centre + float(half))]


def quality(truth: list[str], predictions: list[str | None], labels: list[str]) -> dict[str, Any]:
    if len(truth) != len(predictions) or not truth or not set(truth) <= set(labels):
        raise ValueError("Quality inputs invalid.")
    predicted = [value if value in labels else "__INVALID__" for value in predictions]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "accuracy_wilson_95": wilson(
            sum(a == b for a, b in zip(truth, predicted, strict=True)), len(truth)
        ),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "support": len(truth),
        "invalid_count": predicted.count("__INVALID__"),
        "f1_scope": "77 canonical labels; invalid predictions are false negatives, never dropped",
    }


def execute(
    graph: Any, frame: pd.DataFrame, ledger_path: Path
) -> tuple[list[dict[str, Any]], float]:
    """Persist every completed request before advancing; never export raw support text."""
    if ledger_path.exists():
        raise ValueError("Run ledger exists; choose a new run directory.")
    records: list[dict[str, Any]] = []
    start = perf_counter()
    for identity, row in frame.iterrows():
        state: dict[str, Any] = dict(invoke(graph, str(row["text"])))
        nearest = state["retrieved"]["matches"][0]
        response = state["response"]
        records.append(
            {
                "row_id": int(str(identity)),
                "row_sha256": fingerprint([int(str(identity)), row["text"], row["intent"]]),
                **{
                    key: state[key]
                    for key in (
                        "intent",
                        "baseline_label",
                        "agreement",
                        "needs_review",
                        "route",
                        "cache_hit",
                        "cache_key",
                        "retrieval_seconds",
                        "baseline_seconds",
                        "llm_seconds",
                        "end_to_end_seconds",
                    )
                },
                "nearest_label": nearest["intent"],
                "nearest_similarity": nearest["similarity"],
                "retrieved_ids": [match["id"] for match in state["retrieved"]["matches"]],
                "failure": response.get("failure"),
                "done_reason": response.get("done_reason"),
                **{
                    key: response.get(key)
                    for key in (
                        "prompt_eval_count",
                        "eval_count",
                        "prompt_eval_cached_count",
                        "total_duration",
                        "load_duration",
                        "prompt_eval_duration",
                        "eval_duration",
                    )
                },
            }
        )
        atomic_json(ledger_path, {"rows": records, "elapsed_seconds": perf_counter() - start})
    return records, perf_counter() - start


def summarize_compute(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    groups = {}
    for name, hit in (("live", False), ("cache_hits", True)):
        rows = [row for row in records if row["cache_hit"] == hit]
        groups[name] = {
            "rows": len(rows),
            "latency": {
                stage: latency([row[f"{stage}_seconds"] for row in rows])
                for stage in ("retrieval", "baseline", "llm", "end_to_end")
            },
            "tokens": {
                field: {
                    "known_total": sum(row[field] for row in rows if isinstance(row[field], int)),
                    "unknown_rows": sum(not isinstance(row[field], int) for row in rows),
                }
                for field in ("prompt_eval_count", "eval_count", "prompt_eval_cached_count")
            },
        }
    return {
        "api_cost_usd": 0.0,
        "total_wall_seconds": elapsed,
        "timing_scope": (
            "sequential local batch=1; graph includes identity verification, retrieval, baseline, "
            "LLM HTTP and cache I/O; initialization excluded; first live load is included; "
            "ledger writes included only in total wall time; cache-hit token totals are historical"
        ),
        **groups,
    }


def gate_diagnostics(records: list[dict[str, Any]], truth: list[str]) -> dict[str, Any]:
    candidates = [i for i, row in enumerate(records) if row["agreement"]]
    correct = sum(records[i]["intent"] == truth[i] for i in candidates)
    return {
        "tau": None,
        "target_accepted_accuracy": 0.99,
        "operational_accepted_count": 0,
        "operational_coverage": 0.0,
        "operational_accepted_accuracy": None,
        "note": "Tau PENDING: all rows route to review. Agreement-only candidates are diagnostics.",
        "agreement_only": {
            "support": len(candidates),
            "coverage": len(candidates) / len(records),
            "accuracy": correct / len(candidates) if candidates else None,
            "correct": correct,
            "wilson_95": wilson(correct, len(candidates)),
            "nearest_similarity": {
                "mean": float(np.mean([records[i]["nearest_similarity"] for i in candidates]))
                if candidates
                else None,
                "p95": float(
                    np.percentile([records[i]["nearest_similarity"] for i in candidates], 95)
                )
                if candidates
                else None,
            },
        },
    }


def extrapolation(compute: dict[str, Any]) -> dict[str, Any]:
    timing = compute["live"]["latency"]["end_to_end"]
    return {
        "basis": "live mean/p95; planning range, not a confidence interval; no cache",
        "validation_2001_seconds": [2001 * timing[key] for key in ("mean_seconds", "p95_seconds")]
        if timing["examples"]
        else None,
        "final_test_3080_seconds": [3080 * timing[key] for key in ("mean_seconds", "p95_seconds")]
        if timing["examples"]
        else None,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_json(path.with_suffix(".json"), report)
    lines = [
        "# Phase 2 local experiment",
        "",
        f"Status: **{report['status']}**.",
        "",
        "API cost: **$0.00**. Official test split was not opened. Null means PENDING.",
        "",
    ]
    if "comparison" in report:
        lines += ["| Path | Accuracy | Macro F1 | Weighted F1 |", "| --- | ---: | ---: | ---: |"]
        for name, values in report["comparison"].items():
            lines.append(
                f"| {name} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} "
                f"| {values['weighted_f1']:.6f} |"
            )
        lines.append("")
    if "configuration" in report:
        config = report["configuration"]
        lines += [
            f"Model: `{config['model']}`; manifest `{config['digest']}`.",
            f"Output mechanism: `{config['mechanism']}`; temperature 0, seed 42, cap 48 tokens.",
            f"Split: validation; rows: {report['rows']}; "
            f"represented intents: {report['represented_classes']}.",
            f"Invalid outputs: {report['invalid_output_count']}; "
            f"provider failures: {report['failure_count']}.",
            "",
        ]
    if "compute" in report:
        compute = report["compute"]
        lines += [
            "| Calls | Stage | N | Mean seconds | p95 seconds |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for group in ("live", "cache_hits"):
            for stage, values in compute[group]["latency"].items():
                mean = f"{values['mean_seconds']:.6f}" if values["examples"] else "PENDING"
                p95 = f"{values['p95_seconds']:.6f}" if values["examples"] else "PENDING"
                lines.append(f"| {group} | {stage} | {values['examples']} | {mean} | {p95} |")
        lines += [
            "",
            compute["timing_scope"],
            "",
            f"Graph-loop wall time including ledger writes: {compute['total_wall_seconds']:.3f} s.",
            "",
            "| Calls | Token field | Known total | Unknown rows |",
            "| --- | --- | ---: | ---: |",
        ]
        for group in ("live", "cache_hits"):
            for field, values in compute[group]["tokens"].items():
                lines.append(
                    f"| {group} | {field} | {values['known_total']} | {values['unknown_rows']} |"
                )
        lines.append("")
    if "gate" in report:
        gate = report["gate"]
        diagnostic = gate["agreement_only"]
        lines += [
            "## Gate diagnostics — τ PENDING",
            "",
            "Operational coverage is 0; all rows route to review. Accepted accuracy is PENDING.",
            f"Agreement-only candidate support: {diagnostic['support']}; "
            f"coverage: {diagnostic['coverage']:.4%}; accuracy: {diagnostic['accuracy']}; "
            f"95% Wilson interval: {diagnostic['wilson_95']}.",
            "These candidates do not establish the 99% target or select a similarity threshold.",
            "",
        ]
    if "time_extrapolation" in report:
        projection = report["time_extrapolation"]
        lines += ["## Time planning", "", projection["basis"], ""]
        for key in ("validation_2001_seconds", "final_test_3080_seconds"):
            values = projection[key]
            display = (
                "PENDING"
                if values is None
                else f"{values[0] / 60:.2f}–{values[1] / 60:.2f} minutes"
            )
            lines.append(f"- {key}: {display}.")
        lines += ["", "Neither projected run was performed. Both require later approval.", ""]
    if "limitations" in report:
        lines += ["## Limitations", ""] + [f"- {value}" for value in report["limitations"]] + [""]
    lines += [
        "Machine-readable evidence (including timing, token accounting, row IDs/hashes,",
        f"configuration and limitations) is in [{path.with_suffix('.json').name}]"
        f"({path.with_suffix('.json').name}).",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines))
