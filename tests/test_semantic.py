import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from intentops.semantic import SemanticIndex


def test_mocked_embeddings_round_trip_vote_and_lazy_load(tmp_path: Path) -> None:
    encoder = Mock()
    encoder.encode.return_value = np.asarray([[1, 0], [0.8, 0.2], [0, 1]], dtype=float)
    index = SemanticIndex.build(["one", "two", "three"], ["a", "a", "b"], "mock", "v1", encoder)
    path = tmp_path / "semantic.npz"
    index.save(path)
    restored = SemanticIndex.load(path, "v1")
    assert restored._encoder is None  # Loading an artifact never downloads a model.
    restored._encoder = encoder
    encoder.encode.return_value = np.asarray([[1, 0]], dtype=float)
    result = restored.query("query", 3)
    assert result.intent == "a"
    assert result.vote == pytest.approx(1)
    assert result.matches[0].similarity == pytest.approx(1)
    assert [m.similarity for m in result.matches] == sorted(
        [m.similarity for m in result.matches], reverse=True
    )
    with pytest.raises(ValueError, match="another baseline"):
        SemanticIndex.load(path, "v2")
    with pytest.raises(ValueError, match="positive"):
        restored.query("query", 0)
    encoder.encode.return_value = np.asarray([[0, 0]], dtype=float)
    with pytest.raises(ValueError, match="invalid query"):
        restored.query("query")


def test_zero_vote_still_requires_review_score() -> None:
    encoder = Mock()
    encoder.encode.return_value = np.asarray([[-1, 0]], dtype=float)
    index = SemanticIndex(np.asarray([[1, 0]]), ["one"], ["a"], "mock", "v1", encoder)
    result = index.query("opposite")
    assert result.vote == 0
    assert result.matches[0].similarity == -1


def test_lazy_local_encoder_and_explicit_build(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = Mock()
    encoder.encode.return_value = np.asarray([[1.0, 0]])
    factory = Mock(return_value=encoder)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=factory)
    )
    index = SemanticIndex(np.asarray([[1.0, 0]]), ["public example"], ["a"], "mock", "v1")
    factory.assert_not_called()
    assert index.ensure_encoder() is encoder
    factory.assert_called_once_with("mock", local_files_only=True)
    assert index.ensure_encoder() is encoder
    assert factory.call_count == 1
    factory.reset_mock()
    built = SemanticIndex.build(["public example"], ["a"], "mock", "v1")
    factory.assert_called_once_with("mock")
    assert built.query("query").intent == "a"


def test_missing_optional_dependency_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    index = SemanticIndex(np.asarray([[1.0, 0]]), ["one"], ["a"], "mock", "v1")
    with pytest.raises(RuntimeError, match="Install semantic support"):
        index.ensure_encoder()
    with pytest.raises(RuntimeError, match="Install semantic support"):
        SemanticIndex.build(["one"], ["a"], "mock", "v1")


@pytest.mark.parametrize(
    "embeddings,texts,labels",
    [
        (np.asarray([[0, 0]]), ["one"], ["a"]),
        (np.asarray([[1, 0]]), [], []),
        (np.asarray([[float("nan"), 0]]), ["one"], ["a"]),
    ],
)
def test_invalid_index_rejected(embeddings: np.ndarray, texts: list, labels: list) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        SemanticIndex(embeddings, texts, labels, "mock", "v1")
