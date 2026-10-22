"""Shared API contracts; confidence always means baseline maximum probability."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SupportText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2000)
]
MAX_BATCH_SIZE = 32


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: SupportText


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[SupportText] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class IntentScore(BaseModel):
    intent: str
    probability: float = Field(ge=0, le=1)


class SemanticMatch(BaseModel):
    text: str
    intent: str
    similarity: float = Field(ge=-1, le=1)


class Prediction(BaseModel):
    intent: str
    confidence: float = Field(ge=0, le=1)
    needs_review: bool
    top_k: list[IntentScore]
    model_version: str
    latency_ms: float = Field(ge=0)
    source: Literal["baseline", "semantic"] = "baseline"
    semantic_vote: float | None = None
    semantic_matches: list[SemanticMatch] | None = None


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    detail: str | None = None


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    class_count: int
    review_threshold: float
    semantic_available: bool
    semantic_enabled: bool
    semantic_min_similarity: float
    semantic_min_vote: float
