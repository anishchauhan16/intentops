"""Explicit baseline and optional semantic acceptance rules."""

import logging
from time import perf_counter
from uuid import uuid4

from intentops.config import Settings
from intentops.model import Baseline
from intentops.schemas import Prediction
from intentops.semantic import SemanticIndex

logger = logging.getLogger("intentops.inference")


class Router:
    def __init__(
        self, baseline: Baseline, settings: Settings, semantic: SemanticIndex | None = None
    ) -> None:
        if settings.semantic_enabled and semantic is None:
            raise ValueError(
                "Semantic mode requires semantic.npz. Rebuild using train.py --semantic."
            )
        if semantic is not None and not set(semantic.labels).issubset(baseline.labels):
            raise ValueError("Semantic labels must be present in the baseline model.")
        self.baseline = baseline
        self.settings = settings
        self.semantic = semantic

    def predict(self, text: str) -> Prediction:
        started = perf_counter()
        top = self.baseline.top_k(text, self.settings.top_k)
        prediction = Prediction(
            intent=top[0].intent,
            confidence=top[0].probability,
            needs_review=top[0].probability < self.settings.review_threshold,
            top_k=top,
            model_version=self.baseline.version,
            latency_ms=0,
        )
        if prediction.needs_review and self.settings.semantic_enabled and self.semantic is not None:
            result = self.semantic.query(text)
            prediction.semantic_matches = result.matches
            prediction.semantic_vote = result.vote
            if (
                result.matches
                and result.matches[0].similarity >= self.settings.semantic_min_similarity
                and result.vote >= self.settings.semantic_min_vote
            ):
                prediction.intent = result.intent
                prediction.needs_review = False
                prediction.source = "semantic"
        prediction.latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "prediction",
            extra={
                "event": {
                    "event_id": uuid4().hex,
                    "model_version": prediction.model_version,
                    "intent": prediction.intent,
                    "confidence": prediction.confidence,
                    "needs_review": prediction.needs_review,
                    "latency_ms": prediction.latency_ms,
                    "source": prediction.source,
                }
            },
        )
        return prediction
