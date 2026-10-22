"""Local data validation and deterministic development splitting."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from intentops.config import SEED, VALIDATION_SIZE


def validate_frame(frame: pd.DataFrame) -> None:
    missing = {"text", "intent"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Dataset must contain at least one row.")
    for column in ("text", "intent"):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"Column '{column}' must contain nonempty strings without nulls.")


def read_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Data not found at {path}. Run python scripts/fetch_data.py first."
        )
    frame = pd.read_parquet(path)
    validate_frame(frame)
    return frame


def split_development(
    frame: pd.DataFrame, validation_size: float = VALIDATION_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_frame(frame)
    train, validation = train_test_split(
        frame, test_size=validation_size, stratify=frame["intent"], random_state=seed
    )
    return train.copy(), validation.copy()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
