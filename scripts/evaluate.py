"""Evaluate a frozen model and routing configuration; never tune on test labels."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intentops.config import Settings
from intentops.data import read_data, sha256_file, split_development
from intentops.metrics import (
    classification_metrics,
    confusion_pairs,
    label_metrics,
    selective_accuracy,
)
from intentops.model import Baseline
from intentops.router import Router
from intentops.semantic import SemanticIndex


def evaluate(frame: pd.DataFrame, settings: Settings, output: Path, split: str) -> dict[str, Any]:
    baseline = Baseline.load(settings.artifact_dir)
    truth = frame["intent"].tolist()
    texts = frame["text"].tolist()
    probabilities = baseline.probabilities(texts)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = np.asarray(baseline.labels)[order[:, 0]].tolist()
    confidence = probabilities.max(axis=1).tolist()
    report: dict[str, Any] = {
        "split": split,
        "rows": len(frame),
        "model_version": baseline.version,
        "routing_configuration": {
            "review_threshold": settings.review_threshold,
            "semantic_min_similarity": settings.semantic_min_similarity,
            "semantic_min_vote": settings.semantic_min_vote,
        },
        "baseline": classification_metrics(truth, probabilities, baseline.labels),
        "hybrid": None,
    }
    output.mkdir(parents=True, exist_ok=True)
    selective_accuracy(truth, predicted, confidence).to_csv(
        output / "selective_accuracy.csv", index=False
    )
    confusion_pairs(truth, predicted).to_csv(output / "confusion_pairs.csv", index=False)
    errors = []
    for row, (text, actual, prediction, score) in enumerate(
        zip(texts, truth, predicted, confidence, strict=True)
    ):
        if actual != prediction:
            errors.append(
                {
                    "text": text,
                    "true_intent": actual,
                    "predicted_intent": prediction,
                    "confidence": score,
                    "top_alternatives": json.dumps(
                        [
                            {
                                "intent": baseline.labels[i],
                                "probability": float(probabilities[row, i]),
                            }
                            for i in order[row, :3]
                        ]
                    ),
                }
            )
    pd.DataFrame(
        errors,
        columns=["text", "true_intent", "predicted_intent", "confidence", "top_alternatives"],
    ).to_csv(output / "errors.csv", index=False)
    semantic_path = settings.artifact_dir / "semantic.npz"
    if semantic_path.exists():
        semantic = SemanticIndex.load(semantic_path, baseline.version)
        hybrid_settings = Settings(
            artifact_dir=settings.artifact_dir,
            review_threshold=settings.review_threshold,
            semantic_enabled=True,
            semantic_min_similarity=settings.semantic_min_similarity,
            semantic_min_vote=settings.semantic_min_vote,
        )
        router = Router(baseline, hybrid_settings, semantic)
        routed = [router.predict(text) for text in texts]
        hybrid_predictions = [item.intent for item in routed]
        accepted = np.asarray([not item.needs_review for item in routed])
        correct = np.asarray(truth) == np.asarray(hybrid_predictions)
        hybrid = label_metrics(truth, hybrid_predictions, baseline.labels)
        hybrid.update(
            {
                "coverage": float(accepted.mean()),
                "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
                "number_accepted": int(accepted.sum()),
                "number_review": int((~accepted).sum()),
                "semantic_used": sum(item.semantic_matches is not None for item in routed),
                "semantic_accepted": sum(item.source == "semantic" for item in routed),
                "semantic_model": semantic.model_name,
                "semantic_sha256": sha256_file(semantic_path),
                "accuracy_delta_vs_baseline": hybrid["accuracy"] - report["baseline"]["accuracy"],
                "top_3_accuracy": None,
                "top_3_note": (
                    "Hybrid emits one voted label; top-3 belongs to the baseline distribution."
                ),
            }
        )
        report["hybrid"] = hybrid
        pd.DataFrame(
            [
                {
                    "text": text,
                    "true_intent": actual,
                    "predicted_intent": item.intent,
                    "confidence": item.confidence,
                    "needs_review": item.needs_review,
                    "source": item.source,
                    "semantic_vote": item.semantic_vote,
                }
                for text, actual, item in zip(texts, truth, routed, strict=True)
            ]
        ).to_csv(output / "hybrid_predictions.csv", index=False)
        confusion_pairs(truth, hybrid_predictions).to_csv(
            output / "hybrid_confusion_pairs.csv", index=False
        )
    else:
        for name in ("hybrid_predictions.csv", "hybrid_confusion_pairs.csv"):
            (output / name).unlink(missing_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--review-threshold", type=float, default=0.7)
    parser.add_argument("--semantic-min-similarity", type=float, default=0.65)
    parser.add_argument("--semantic-min-vote", type=float, default=0.7)
    args = parser.parse_args()
    try:
        baseline = Baseline.load(args.artifact_dir)
        if args.split == "validation":
            config = baseline.metadata["configuration"]
            _, frame = split_development(
                read_data(args.data_dir / "train.parquet"),
                config["validation_size"],
                config["seed"],
            )
        else:
            frame = read_data(args.data_dir / "test.parquet")
        settings = Settings(
            artifact_dir=args.artifact_dir,
            review_threshold=args.review_threshold,
            semantic_min_similarity=args.semantic_min_similarity,
            semantic_min_vote=args.semantic_min_vote,
        )
        output = args.output_dir or (
            Path("reports") if args.split == "test" else Path("reports/validation")
        )
        report = evaluate(frame, settings, output, args.split)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(
        json.dumps(
            {
                "split": args.split,
                "baseline_accuracy": report["baseline"]["accuracy"],
                "hybrid_accuracy": report["hybrid"]["accuracy"] if report["hybrid"] else None,
                "reports": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
