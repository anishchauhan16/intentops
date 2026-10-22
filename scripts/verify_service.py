"""Verify a local or approved public release using fictional requests only."""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


def verify_service(base_url: str, expected_version: str) -> dict[str, Any]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise ValueError("Use an HTTP(S) service origin without embedded credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Use the service origin without a path, query, or fragment.")
    origin = base_url.rstrip("/")
    checks: list[str] = []

    def request(
        path: str,
        data: bytes | None = None,
        content_type: str = "application/json",
        expected_status: int = 200,
    ) -> tuple[str, Any]:
        req = Request(origin + path, data=data, headers={"Content-Type": content_type})
        try:
            with urlopen(req, timeout=120) as response:
                status, body, headers = response.status, response.read(), response.headers
        except HTTPError as exc:
            status, body, headers = exc.code, exc.read(), exc.headers
        if status != expected_status:
            raise ValueError(f"{path}: expected HTTP {expected_status}, received {status}.")
        return body.decode("utf-8"), headers

    def post(text: str) -> dict[str, Any]:
        body, _ = request("/predict", json.dumps({"text": text}).encode())
        prediction: dict[str, Any] = json.loads(body)
        if prediction.get("model_version") != expected_version:
            raise ValueError("Prediction model version differs from the approved release.")
        return prediction

    health = json.loads(request("/health")[0])
    if health.get("status") != "ok" or health.get("model_loaded") is not True:
        raise ValueError("Service does not report a loaded, healthy model.")
    checks.append("health")
    metadata = json.loads(request("/metadata")[0])
    if (
        metadata.get("model_version") != expected_version
        or metadata.get("class_count") != 77
        or metadata.get("review_threshold") != 0.7
        or metadata.get("semantic_enabled") is not False
    ):
        raise ValueError("Metadata differs from the approved 77-class baseline demo configuration.")
    checks.append("metadata")
    accepted = post("How do I change my PIN?")
    if accepted["intent"] != "change_pin" or accepted["needs_review"] is not False:
        raise ValueError("The frozen PIN example did not produce the expected accepted intent.")
    probabilities = [item["probability"] for item in accepted["top_k"]]
    if (
        probabilities != sorted(probabilities, reverse=True)
        or accepted["confidence"] != probabilities[0]
    ):
        raise ValueError("Classifier alternatives or confidence are inconsistent.")
    checks.append("accepted_prediction")
    for text in ("I am still waiting for my card", "I need help"):
        if post(text)["needs_review"] is not True:
            raise ValueError("A review example unexpectedly bypassed the configured review gate.")
    checks.append("uncertain_and_ambiguous_review")
    request("/predict", json.dumps({"text": ""}).encode(), expected_status=422)
    request("/batch-predict", json.dumps({"texts": ["hello"] * 33}).encode(), expected_status=422)
    batch = json.loads(
        request("/batch-predict", json.dumps({"texts": ["reset my pin"]}).encode())[0]
    )
    if len(batch["predictions"]) != 1:
        raise ValueError("Batch response does not contain the expected number of predictions.")
    checks.append("validation_and_batch")
    page, headers = request("/")
    if "Route a request" not in page or "form-action 'self'" not in headers.get(
        "Content-Security-Policy", ""
    ):
        raise ValueError("Demo form or content security policy is missing.")
    request("/static/demo.css")
    request("/docs")
    for text, expected in (
        ("How do I change my PIN?", "Automatically accepted"),
        ("I am still waiting for my card", "Human review"),
    ):
        body, headers = request(
            "/demo", urlencode({"text": text}).encode(), "application/x-www-form-urlencoded"
        )
        if expected not in body or headers.get("Cache-Control") != "no-store":
            raise ValueError("Demo decision or no-store response policy is missing.")
    checks.append("browser_form_assets_docs_and_decisions")
    return {
        "status": "passed",
        "base_url": origin,
        "model_version": expected_version,
        "checks": checks,
        "scope": "functional HTTP verification; not a latency/load benchmark",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/service_verification.json"))
    args = parser.parse_args()
    try:
        report = verify_service(args.base_url, args.expected_version)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Service verification failed: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
