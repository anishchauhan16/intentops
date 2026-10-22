"""Metrics shared by validation and held-out evaluation."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from intentops.config import THRESHOLDS


def classification_metrics(
    truth: Sequence[str], probabilities: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    if len(truth) == 0 or probabilities.shape != (len(truth), len(labels)):
        raise ValueError(
            "Expected nonempty truth and a probability matrix matching rows and labels."
        )
    if not set(truth).issubset(labels):
        raise ValueError("Evaluation contains labels absent from the training model.")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    predicted = np.asarray(labels)[order[:, 0]].tolist()
    top_three = np.asarray(labels)[order[:, :3]]
    return {
        **label_metrics(truth, predicted, labels),
        "top_3_accuracy": float(np.mean(np.any(top_three == np.asarray(truth)[:, None], axis=1))),
    }


def label_metrics(
    truth: Sequence[str], predicted: Sequence[str], labels: list[str]
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": labels,
            "rows_true_columns_predicted": confusion_matrix(
                truth, predicted, labels=labels
            ).tolist(),
        },
    }


def selective_accuracy(
    truth: Sequence[str],
    predicted: Sequence[str],
    confidence: Sequence[float],
    thresholds: Sequence[float] = THRESHOLDS,
) -> pd.DataFrame:
    if not (len(truth) == len(predicted) == len(confidence)) or len(truth) == 0:
        raise ValueError("Truth, predictions, and confidence must have equal, nonzero lengths.")
    scores = np.asarray(confidence, dtype=float)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Confidence must be finite and between 0 and 1.")
    correct = np.asarray(truth) == np.asarray(predicted)
    rows = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        accepted = scores >= threshold
        count = int(accepted.sum())
        rows.append(
            {
                "threshold": threshold,
                "coverage": count / len(truth),
                "accepted_accuracy": float(correct[accepted].mean()) if count else None,
                "number_accepted": count,
                "number_review": len(truth) - count,
            }
        )
    return pd.DataFrame(rows)


def confusion_pairs(truth: Sequence[str], predicted: Sequence[str]) -> pd.DataFrame:
    counts = Counter((a, b) for a, b in zip(truth, predicted, strict=True) if a != b)
    rows = [
        {"true_intent": a, "predicted_intent": b, "count": count}
        for (a, b), count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return pd.DataFrame(rows, columns=["true_intent", "predicted_intent", "count"])
