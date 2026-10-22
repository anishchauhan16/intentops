import io
import json
import logging

from intentops.logging import JsonFormatter
from intentops.router import Router


def test_inference_event_has_no_raw_text(router: Router) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("intentops.inference")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        router.predict("private account details 12345")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    event = json.loads(stream.getvalue())
    assert "private account" not in stream.getvalue()
    assert {
        "event_id",
        "model_version",
        "intent",
        "confidence",
        "needs_review",
        "latency_ms",
    } <= event.keys()
