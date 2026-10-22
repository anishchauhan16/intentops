import numpy as np
import pytest

from intentops.metrics import classification_metrics, confusion_pairs, selective_accuracy


def test_selective_accuracy_exact_boundary_and_empty_acceptance() -> None:
    table = selective_accuracy(
        ["a", "b", "a", "b"], ["a", "a", "a", "b"], [0.9, 0.8, 0.6, 0.5], [0.5, 0.8, 0.95]
    )
    assert table.iloc[0]["coverage"] == 1
    assert table.iloc[0]["accepted_accuracy"] == 0.75
    assert table.iloc[1]["coverage"] == 0.5
    assert table.iloc[1]["accepted_accuracy"] == 0.5
    assert table.iloc[1]["number_accepted"] == 2
    assert table.iloc[1]["number_review"] == 2
    assert table.iloc[2]["number_accepted"] == 0
    assert np.isnan(table.iloc[2]["accepted_accuracy"])


@pytest.mark.parametrize(
    "truth,predicted,scores,thresholds",
    [
        ([], [], [], [0.5]),
        (["a"], [], [0.7], [0.5]),
        (["a"], ["a"], [float("nan")], [0.5]),
        (["a"], ["a"], [1.1], [0.5]),
        (["a"], ["a"], [0.5], [-0.1]),
    ],
)
def test_selective_accuracy_rejects_invalid_inputs(
    truth: list,
    predicted: list,
    scores: list,
    thresholds: list,
) -> None:
    with pytest.raises(ValueError):
        selective_accuracy(truth, predicted, scores, thresholds)


def test_known_metrics_top_three_and_matrix() -> None:
    labels = ["a", "b", "c", "d"]
    probabilities = np.asarray([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    result = classification_metrics(["b", "a"], probabilities, labels)
    assert result["accuracy"] == 0
    assert result["macro_f1"] == 0
    assert result["top_3_accuracy"] == 0.5
    assert result["confusion_matrix"]["rows_true_columns_predicted"][1][0] == 1
    perfect = classification_metrics(["a", "d"], probabilities, labels)
    assert perfect["accuracy"] == 1
    assert perfect["weighted_f1"] == 1
    with pytest.raises(ValueError, match="probability matrix"):
        classification_metrics([], probabilities, labels)
    with pytest.raises(ValueError, match="absent"):
        classification_metrics(["x", "a"], probabilities, labels)


def test_confusion_pairs_are_directed_sorted_and_have_empty_schema() -> None:
    pairs = confusion_pairs(["a", "a", "b", "c"], ["b", "b", "a", "c"])
    assert pairs.to_dict("records") == [
        {"true_intent": "a", "predicted_intent": "b", "count": 2},
        {"true_intent": "b", "predicted_intent": "a", "count": 1},
    ]
    assert list(confusion_pairs(["a"], ["a"]).columns) == [
        "true_intent",
        "predicted_intent",
        "count",
    ]
