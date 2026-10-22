from unittest.mock import Mock

import pytest

from intentops.config import Settings
from intentops.model import Baseline
from intentops.router import Router
from intentops.schemas import IntentScore, SemanticMatch
from intentops.semantic import SemanticIndex, SemanticResult


@pytest.mark.parametrize("confidence,review", [(0.2, True), (0.95, False), (0.7, False)])
def test_review_threshold_mocked(confidence: float, review: bool) -> None:
    baseline = Mock(spec=Baseline, version="mock-v1", labels=["card_arrival"])
    baseline.top_k.return_value = [IntentScore(intent="card_arrival", probability=confidence)]
    result = Router(baseline, Settings(review_threshold=0.7)).predict("private support text")
    assert result.needs_review is review
    assert result.semantic_matches is None
    assert result.latency_ms >= 0


@pytest.mark.parametrize(
    "similarity,vote,accepted",
    [
        (0.9, 0.8, True),
        (0.4, 0.9, False),
        (0.9, 0.4, False),
    ],
)
def test_semantic_acceptance_keeps_classifier_confidence(
    similarity: float,
    vote: float,
    accepted: bool,
) -> None:
    baseline = Mock(spec=Baseline, version="mock-v1", labels=["card_arrival", "change_pin"])
    baseline.top_k.return_value = [IntentScore(intent="card_arrival", probability=0.2)]
    semantic = Mock(spec=SemanticIndex, labels=["change_pin"])
    semantic.query.return_value = SemanticResult(
        "change_pin",
        vote,
        [
            SemanticMatch(
                text="a public training example", intent="change_pin", similarity=similarity
            )
        ],
    )
    result = Router(baseline, Settings(semantic_enabled=True), semantic).predict("reset pin")
    assert result.needs_review is not accepted
    assert result.intent == ("change_pin" if accepted else "card_arrival")
    assert result.source == ("semantic" if accepted else "baseline")
    assert result.confidence == 0.2
    assert result.semantic_matches is not None


def test_semantic_not_used_when_baseline_accepted() -> None:
    baseline = Mock(spec=Baseline, version="mock-v1", labels=["change_pin"])
    baseline.top_k.return_value = [IntentScore(intent="change_pin", probability=0.99)]
    semantic = Mock(spec=SemanticIndex, labels=["change_pin"])
    result = Router(baseline, Settings(semantic_enabled=True), semantic).predict("reset pin")
    semantic.query.assert_not_called()
    assert result.semantic_matches is None
    with pytest.raises(ValueError, match="requires semantic.npz"):
        Router(baseline, Settings(semantic_enabled=True))
    semantic.labels = ["unknown"]
    with pytest.raises(ValueError, match="Semantic labels"):
        Router(baseline, Settings(), semantic)


@pytest.mark.parametrize(
    "kwargs", [{"review_threshold": 1.1}, {"semantic_min_vote": -1}, {"top_k": 0}]
)
def test_settings_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTOPS_REVIEW_THRESHOLD", "0.85")
    monkeypatch.setenv("INTENTOPS_SEMANTIC_ENABLED", "true")
    settings = Settings.from_env()
    assert settings.review_threshold == 0.85
    assert settings.semantic_enabled
