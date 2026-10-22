"""Optional cosine retrieval. Importing this module never loads or downloads a model."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from intentops.schemas import SemanticMatch


@dataclass(frozen=True)
class SemanticResult:
    intent: str
    vote: float
    matches: list[SemanticMatch]


class SemanticIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        labels: list[str],
        model_name: str,
        baseline_version: str,
        encoder: Any = None,
    ) -> None:
        if (
            embeddings.ndim != 2
            or len(texts) == 0
            or embeddings.shape[0] != len(texts)
            or len(labels) != len(texts)
            or embeddings.shape[1] == 0
            or not np.isfinite(embeddings).all()
            or np.any(np.linalg.norm(embeddings, axis=1) == 0)
        ):
            raise ValueError("Semantic embeddings must be finite, nonzero, and match texts/labels.")
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.texts = texts
        self.labels = labels
        self.model_name = model_name
        self.baseline_version = baseline_version
        self._encoder = encoder

    def ensure_encoder(self) -> Any:
        """Load the encoder from local cache only, or return an injected encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    'Install semantic support: pip install -e ".[semantic]"'
                ) from exc
            self._encoder = SentenceTransformer(self.model_name, local_files_only=True)
        return self._encoder

    @classmethod
    def build(
        cls,
        texts: list[str],
        labels: list[str],
        model_name: str,
        baseline_version: str,
        encoder: Any = None,
    ) -> "SemanticIndex":
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    'Install semantic support: pip install -e ".[semantic]"'
                ) from exc
            encoder = SentenceTransformer(model_name)
        embeddings = np.asarray(
            encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )
        return cls(embeddings, texts, labels, model_name, baseline_version, encoder)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            texts=np.asarray(self.texts),
            labels=np.asarray(self.labels),
            model_name=np.asarray(self.model_name),
            baseline_version=np.asarray(self.baseline_version),
        )

    @classmethod
    def load(cls, path: Path, baseline_version: str) -> "SemanticIndex":
        with np.load(path, allow_pickle=False) as data:
            stored_version = str(data["baseline_version"].item())
            if stored_version != baseline_version:
                raise ValueError(
                    "Semantic artifact belongs to another baseline; rebuild with --semantic."
                )
            return cls(
                data["embeddings"],
                data["texts"].tolist(),
                data["labels"].tolist(),
                str(data["model_name"].item()),
                stored_version,
            )

    def query(self, text: str, k: int = 5) -> SemanticResult:
        if k < 1:
            raise ValueError("k must be positive.")
        query = np.asarray(
            self.ensure_encoder().encode([text], normalize_embeddings=True, show_progress_bar=False)
        )[0]
        norm = np.linalg.norm(query)
        if not np.isfinite(query).all() or norm == 0:
            raise ValueError("Semantic encoder returned an invalid query embedding.")
        similarities = np.clip(self.embeddings @ (query / norm), -1, 1)
        indices = np.argsort(-similarities, kind="stable")[:k]
        matches = [
            SemanticMatch(
                text=self.texts[i], intent=self.labels[i], similarity=float(similarities[i])
            )
            for i in indices
        ]
        votes: dict[str, float] = defaultdict(float)
        for match in matches:
            votes[match.intent] += max(match.similarity, 0.0)
        winner = sorted(votes, key=lambda label: (-votes[label], label))[0]
        total = sum(votes.values())
        return SemanticResult(winner, votes[winner] / total if total else 0.0, matches)
