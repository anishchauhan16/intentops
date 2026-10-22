"""Training defaults and environment-backed service settings."""

import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
VALIDATION_SIZE = 0.2
DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = SEED
    validation_size: float = VALIDATION_SIZE
    min_df: int = 2
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    c: float = 4.0
    max_iter: int = 1000


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path = Path("artifacts")
    review_threshold: float = 0.7
    semantic_enabled: bool = False
    semantic_min_similarity: float = 0.65
    semantic_min_vote: float = 0.7
    top_k: int = 3

    def __post_init__(self) -> None:
        for value in (self.review_threshold, self.semantic_min_similarity, self.semantic_min_vote):
            if not 0 <= value <= 1:
                raise ValueError("Routing thresholds must be between 0 and 1.")
        if self.top_k < 1:
            raise ValueError("top_k must be positive.")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            artifact_dir=Path(os.getenv("INTENTOPS_ARTIFACT_DIR", "artifacts")),
            review_threshold=float(os.getenv("INTENTOPS_REVIEW_THRESHOLD", "0.7")),
            semantic_enabled=os.getenv("INTENTOPS_SEMANTIC_ENABLED", "false").lower() == "true",
            semantic_min_similarity=float(os.getenv("INTENTOPS_SEMANTIC_MIN_SIMILARITY", "0.65")),
            semantic_min_vote=float(os.getenv("INTENTOPS_SEMANTIC_MIN_VOTE", "0.7")),
        )
