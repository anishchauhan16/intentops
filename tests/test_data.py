from pathlib import Path

import pandas as pd
import pytest

from intentops.data import read_data, sha256_file, split_development, validate_frame


def test_missing_columns_rejected() -> None:
    with pytest.raises(ValueError, match="Missing required columns: intent"):
        validate_frame(pd.DataFrame({"text": ["hello"]}))


@pytest.mark.parametrize("text", ["", "  ", None, 123])
def test_invalid_text_rejected(text: object) -> None:
    with pytest.raises(ValueError, match="nonempty strings"):
        validate_frame(pd.DataFrame({"text": [text], "intent": ["card_arrival"]}))


def test_empty_rejected() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        validate_frame(pd.DataFrame(columns=["text", "intent"]))


def test_split_deterministic_and_disjoint(development: pd.DataFrame) -> None:
    train, validation = split_development(development)
    second_train, second_validation = split_development(development)
    pd.testing.assert_frame_equal(train, second_train)
    pd.testing.assert_frame_equal(validation, second_validation)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index) | set(validation.index) == set(development.index)
    assert set(train.intent) == set(validation.intent) == set(development.intent)


def test_parquet_round_trip_and_checksum(tmp_path: Path, development: pd.DataFrame) -> None:
    path = tmp_path / "train.parquet"
    development.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(read_data(path), development)
    assert len(sha256_file(path)) == 64
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        read_data(tmp_path / "missing.parquet")
