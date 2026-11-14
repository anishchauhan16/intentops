"""Predeclared threshold selection, immutable freeze and one-shot local evaluation."""

import ast
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from intentops.data import sha256_file, validate_frame
from intentops.llm_cache import ResponseCache, atomic_json, fingerprint
from intentops.llm_evaluation import summarize_compute, wilson
from intentops.llm_path import LocalConfig, build_graph, output_schema, routing_gate

RULE = {
    "eligibility": "accepted correct/support >= 0.99 AND support >= 400",
    "eligible_order": "maximum coverage, then lower tau",
    "fallback_order": "maximum accuracy, then maximum coverage, then lower tau; target MISSED",
    "grid": "unique agreement similarities plus nextafter(minimum,-inf); strict similarity > tau",
    "empty_agreement": "tau=1.0; zero accepted; accuracy null; target MISSED",
}
SOURCE_PATHS = (
    "src/intentops/llm_path.py",
    "src/intentops/llm_cache.py",
    "src/intentops/llm_evaluation.py",
    "src/intentops/llm_final.py",
    "scripts/finalize_llm.py",
    "configs/llm-pilot.json",
    "configs/intent-descriptions-v1.json",
    "src/intentops/model.py",
    "src/intentops/data.py",
    "src/intentops/config.py",
    "src/intentops/transformer_model.py",
    "pyproject.toml",
)


def prompt_contract(
    config: LocalConfig, labels: list[str], source: str | None = None
) -> dict[str, str]:
    """Hash exact prompt-building AST, schema and canonical label order, excluding tau."""
    tree = ast.parse(source if source is not None else inspect.getsource(build_graph))
    classify = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "classify"
    )
    assignments = [
        node
        for node in classify.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in {"instruction", "messages"}
            for target in node.targets
        )
    ]
    if len(assignments) != 2:
        raise ValueError("Frozen prompt assignments could not be identified.")
    return {
        "prompt_sha256": fingerprint(
            {
                "ast": [ast.dump(node, include_attributes=False) for node in assignments],
                "labels": labels,
                "mechanism": config.mechanism,
                "prompt_version": config.prompt_version,
            }
        ),
        "schema_sha256": fingerprint(output_schema(labels)),
    }


def gate_metrics(records: list[dict[str, Any]], truth: list[str], tau: float) -> dict[str, Any]:
    if len(records) != len(truth) or not records:
        raise ValueError("Gate inputs must be aligned and nonempty.")
    accepted = [
        i
        for i, row in enumerate(records)
        if not routing_gate(
            row["intent"],
            row["baseline_label"],
            row["nearest_label"],
            row["nearest_similarity"],
            tau,
        )["needs_review"]
    ]
    correct = sum(records[i]["intent"] == truth[i] for i in accepted)
    count = len(accepted)
    return {
        "tau": tau,
        "support": count,
        "correct": correct,
        "review_support": len(records) - count,
        "coverage": count / len(records),
        "accepted_accuracy": correct / count if count else None,
        "wilson_95": wilson(correct, count),
    }


def select_tau(records: list[dict[str, Any]], truth: list[str]) -> dict[str, Any]:
    similarities = sorted(
        {
            float(row["nearest_similarity"])
            for row in records
            if row["intent"] is not None
            and row["intent"] == row["baseline_label"] == row["nearest_label"]
        }
    )
    if any(not math.isfinite(value) for value in similarities):
        raise ValueError("Nonfinite similarity cannot be selected.")
    grid = [math.nextafter(similarities[0], -math.inf), *similarities] if similarities else [1.0]
    sweep = [gate_metrics(records, truth, tau) for tau in grid]
    eligible = [
        row
        for row in sweep
        if row["support"] >= 400 and row["correct"] * 100 >= 99 * row["support"]
    ]
    candidates = eligible or [row for row in sweep if row["support"]]
    if eligible:
        chosen = min(candidates, key=lambda row: (-row["support"], row["tau"]))
    elif candidates:
        chosen = min(
            candidates, key=lambda row: (-row["accepted_accuracy"], -row["support"], row["tau"])
        )
    else:
        chosen = sweep[0]
    return {
        "rule": RULE,
        "target_status": "MET" if eligible else "MISSED",
        "eligible_thresholds": len(eligible),
        "chosen": chosen,
        "sweep": sweep,
    }


class CacheOnly(ResponseCache):
    def resolve(self, key: str, create: Any) -> tuple[dict[str, Any], bool]:
        def missing() -> dict[str, Any]:
            raise ValueError(
                "Pilot cache miss; stop before issuing an unapproved replacement call."
            )

        return super().resolve(key, missing)


def source_identity() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *SOURCE_PATHS], text=True
    ).strip()
    if status:
        raise ValueError("Commit the evaluator and frozen inputs before starting evaluation.")
    return {
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: sha256_file(Path(name)) for name in SOURCE_PATHS},
    }


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def create_freeze(
    path: Path,
    config: LocalConfig,
    labels: list[str],
    binding: dict[str, Any],
    selection: dict[str, Any],
    validation_path: Path,
    model_identity: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "created_at": datetime.now(UTC).isoformat(),
        **source_identity(),
        "pilot_configuration": asdict(config),
        "configuration": {**asdict(config), "tau": selection["chosen"]["tau"]},
        **prompt_contract(config, labels),
        "ordered_labels_sha256": fingerprint(labels),
        "index_binding": binding,
        "models": model_identity,
        "selection": {key: value for key, value in selection.items() if key != "sweep"},
        "validation_report_sha256": sha256_file(validation_path),
        "note": "Written and fsynced before any official test file access.",
    }
    exclusive_json(path, record)
    return record


def claim_test(
    receipt_path: Path,
    freeze_path: Path,
    config: LocalConfig,
    labels: list[str],
    binding: dict[str, Any],
    validation_path: Path,
    model_identity: dict[str, Any],
) -> dict[str, Any]:
    frozen = json.loads(freeze_path.read_text())
    if (
        frozen["pilot_configuration"] != asdict(config)
        or frozen["index_binding"] != binding
        or frozen["models"] != model_identity
        or frozen["validation_report_sha256"] != sha256_file(validation_path)
        or any(frozen[key] != value for key, value in prompt_contract(config, labels).items())
        or any(frozen[key] != value for key, value in source_identity().items())
    ):
        raise ValueError("Freeze receipt no longer matches; test must remain unopened.")
    exclusive_json(
        receipt_path,
        {
            "status": "started",
            "started_at": datetime.now(UTC).isoformat(),
            "freeze_sha256": sha256_file(freeze_path),
            "code_commit": frozen["code_commit"],
            "note": "One-shot attempt claimed BEFORE test access; no automatic retry.",
        },
    )
    return frozen


def read_final_test(path: Path, expected_sha256: str, labels: list[str]) -> pd.DataFrame:
    """One file read after claim; use the same loaded frame for all three frozen paths."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Official test checksum mismatch; consumed attempt must be reviewed.")
    frame = pd.read_parquet(io.BytesIO(raw))
    validate_frame(frame)
    if len(frame) != 3080 or set(frame.intent) != set(labels):
        raise ValueError("Expected every official test row and all 77 canonical labels.")
    return frame


def summarize_run(
    split: str,
    frame: pd.DataFrame,
    records: list[dict[str, Any]],
    elapsed: float,
    comparisons: dict[str, Any],
    tau: float,
) -> dict[str, Any]:
    if [int(i) for i in frame.index] != [row["row_id"] for row in records]:
        raise ValueError("Evaluation records do not match the exact ordered example set.")
    return {
        "split": split,
        "rows": len(frame),
        "represented_classes": frame.intent.nunique(),
        "ordered_rows_sha256": fingerprint([row["row_sha256"] for row in records]),
        "comparison": comparisons,
        "gate": gate_metrics(records, frame.intent.tolist(), tau),
        "invalid_count": sum(row["intent"] is None for row in records),
        "failure_count": sum(bool(row["failure"]) for row in records),
        "compute": summarize_compute(records, elapsed),
    }


def save_comparison(path: Path, document: dict[str, Any]) -> None:
    atomic_json(path.with_suffix(".json"), document)
    lines = [
        "# Local LLM/RAG versus frozen baseline and transformer",
        "",
        "Single final test evaluation; configuration frozen before test access. API cost $0.00.",
        "",
    ]
    for split in ("validation", "test"):
        report = document[split]
        gate = report["gate"]
        lines += [
            f"## {split.title()} — {report['rows']} rows",
            "",
            "| Path | Accuracy | Macro F1 | Weighted F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, values in report["comparison"].items():
            lines.append(
                f"| {name} | {values['accuracy']:.6%} | {values['macro_f1']:.6f} "
                f"| {values['weighted_f1']:.6f} |"
            )
        lines += [
            "",
            f"Frozen τ: `{gate['tau']!r}`. Gate support: {gate['support']}; "
            f"coverage: {gate['coverage']:.4%}; accepted accuracy: {gate['accepted_accuracy']}; "
            f"Wilson 95%: {gate['wilson_95']}.",
            f"Invalid outputs: {report['invalid_count']}; failures: {report['failure_count']}.",
            "",
            "| Calls | Stage | N | Mean seconds | p95 seconds |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for group in ("live", "cache_hits"):
            for stage, values in report["compute"][group]["latency"].items():
                lines.append(
                    f"| {group} | {stage} | {values['examples']} "
                    f"| {values['mean_seconds']} | {values['p95_seconds']} |"
                )
        lines += [
            "",
            "| Calls | Token field | Known total | Unknown rows |",
            "| --- | --- | ---: | ---: |",
        ]
        for group in ("live", "cache_hits"):
            for field, value in report["compute"][group]["tokens"].items():
                lines.append(
                    f"| {group} | {field} | {value['known_total']} | {value['unknown_rows']} |"
                )
        lines += [
            "",
            f"Graph-loop wall time: {report['compute']['total_wall_seconds']:.3f} seconds.",
            report["compute"]["timing_scope"],
            "",
        ]
    lines += [
        f"Validation target: **{document['selection']['target_status']}** under the declared rule.",
        f"Published baseline/transformer reproduction: {document['test']['reproduction']}.",
        "",
        "Validation selects only τ. Test results do not select any model, prompt or threshold.",
        "Wilson intervals describe observed support; they do not guarantee deployment accuracy.",
        "Local timings differ from hosted API latency and Phase 1 CPU benchmarks.",
        "No retraining, serving changes, deployment, push or merge occurred.",
        "",
        f"[Machine-readable evidence]({path.with_suffix('.json').name})",
        "",
    ]
    path.with_suffix(".md").write_text("\n".join(lines))
