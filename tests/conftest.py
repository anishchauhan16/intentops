"""Shared synthetic fixtures. Any attempted network connection fails immediately."""

import socket
from pathlib import Path

import pandas as pd
import pytest

from intentops.config import Settings, TrainingConfig
from intentops.model import Baseline, train_baseline
from intentops.router import Router


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unit tests must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(scope="session")
def development() -> pd.DataFrame:
    examples = {
        "card_arrival": [
            "waiting for my card delivery",
            "where is my new card",
            "card delivery is delayed",
            "my card has not arrived",
            "track card delivery please",
            "still waiting on the card",
            "card delivery tracking number",
            "new bank card is late",
            "when will my card arrive",
            "missing new card delivery",
        ],
        "change_pin": [
            "change my card pin",
            "reset the pin code",
            "how to change pin",
            "need a new pin code",
            "update my pin number",
            "replace my old pin",
            "set another pin code",
            "forgot pin number",
            "choose new pin please",
            "help me reset pin",
        ],
        "transfer_timing": [
            "bank transfer is delayed",
            "when will transfer arrive",
            "transfer processing time",
            "how long for bank transfer",
            "waiting for money transfer",
            "pending transfer timing",
            "my bank transfer is late",
            "money transfer arrival time",
            "transfer has not arrived",
            "why is transfer taking days",
        ],
    }
    return pd.DataFrame(
        [{"text": text, "intent": label} for label, texts in examples.items() for text in texts]
    )


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory, development: pd.DataFrame) -> Path:
    directory = tmp_path_factory.mktemp("artifacts")
    train_baseline(development, directory, TrainingConfig(min_df=1))
    return directory


@pytest.fixture
def baseline(artifact_dir: Path) -> Baseline:
    return Baseline.load(artifact_dir)


@pytest.fixture
def router(baseline: Baseline) -> Router:
    return Router(baseline, Settings(review_threshold=0.5))
