"""Synthetic-only local-path tests; no network, server, dataset or model weights."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from intentops.llm_cache import ResponseCache, atomic_json, fingerprint
from intentops.llm_evaluation import (
    execute,
    extrapolation,
    gate_diagnostics,
    quality,
    stratified_sample,
    summarize_compute,
    wilson,
    write_report,
)
from intentops.llm_path import (
    MODEL_DIGEST,
    MODEL_TAG,
    ChromaRetriever,
    LocalConfig,
    LocalEncoder,
    OllamaProvider,
    build_graph,
    index_binding,
    load_config,
    output_schema,
    routing_gate,
    validate_output,
)

LABELS = [f"synthetic_{i}" for i in range(77)]


def response(content: str, **changes: Any) -> dict[str, Any]:
    return {
        "content": content,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
        "prompt_eval_cached_count": 0,
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "different"},
        {"digest": "a" * 64},
        {"seed": 1},
        {"temperature": 1},
        {"embedding_revision": "main"},
        {"num_predict": 1000},
        {"tau": 0.5},
        {"mechanism": "other"},
    ],
)
def test_pinned_config_rejects_substitution(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LocalConfig(**changes)


@pytest.mark.parametrize(
    "content",
    [
        '"synthetic_0"',
        '{"intent":"synthetic_0","extra":1}',
        '{"intent":"unknown"}',
        '{"intent":"synthetic_0","intent":"synthetic_1"}',
        '{"intent":null}',
        '{"intent":["synthetic_0"]}',
        '```json\n{"intent":"synthetic_0"}\n```',
        '{"intent":',
    ],
)
def test_strict_output_invalid_policy(content: str) -> None:
    assert validate_output(response(content), LABELS, "json_schema") is None


def test_output_schema_truncation_and_parser() -> None:
    schema = output_schema(LABELS)
    assert schema["properties"]["intent"]["enum"] == LABELS
    assert schema["additionalProperties"] is False
    with pytest.raises(ValueError):
        output_schema(LABELS[:-1])
    assert validate_output(response('{"intent":"synthetic_0"}'), LABELS, "json_schema") == LABELS[0]
    assert validate_output(response("{}", done_reason="length"), LABELS, "json_schema") is None
    assert validate_output(response("{}", done=False), LABELS, "json_schema") is None
    assert validate_output(response("{}", failure="timeout"), LABELS, "json_schema") is None
    assert validate_output(response(None), LABELS, "json_schema") is None
    assert validate_output(response(LABELS[0]), LABELS, "strict_parser") == LABELS[0]
    assert validate_output(response(" " + LABELS[0]), LABELS, "strict_parser") is None


def test_cache_atomic_integrity_and_key_scope(tmp_path: Path) -> None:
    pytest.importorskip("filelock")
    cache = ResponseCache(tmp_path)
    calls = []

    def create() -> dict[str, Any]:
        calls.append(1)
        return response('{"intent":"synthetic_0"}')

    key = fingerprint({"model": MODEL_DIGEST, "text": "private synthetic request"})
    assert cache.resolve(key, create)[1] is False
    assert cache.resolve(key, create)[1] is True
    assert len(calls) == 1
    assert not list(tmp_path.glob(".tmp-*"))
    stored = json.loads((tmp_path / f"{key}.json").read_text())
    stored["response"]["content"] = "changed"
    atomic_json(tmp_path / f"{key}.json", stored)
    with pytest.raises(ValueError, match="checksum"):
        cache.resolve(key, create)
    with pytest.raises(ValueError, match="SHA-256"):
        cache.resolve("../escape", create)
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_sampling_quality_and_uncertainty() -> None:
    frame = pd.DataFrame({"text": ["synthetic"] * 770, "intent": LABELS * 10})
    sample = stratified_sample(frame, 250)
    assert len(sample) == 250 and sample.intent.nunique() == 77
    assert sample.index.tolist() == stratified_sample(frame, 250).index.tolist()
    assert len(stratified_sample(frame, 20)) == 20
    with pytest.raises(ValueError):
        stratified_sample(frame, 1000)
    metrics = quality([LABELS[0], LABELS[1]], [LABELS[0], None], LABELS)
    assert metrics["accuracy"] == 0.5 and metrics["weighted_f1"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(1 / 77)
    assert metrics["invalid_count"] == 1 and metrics["support"] == 2
    with pytest.raises(ValueError):
        quality([], [], LABELS)
    assert wilson(0, 0) is None
    interval = wilson(1, 1)
    assert interval is not None and interval[0] < 0.99 and interval[1] == pytest.approx(1)


class FakeProvider:
    version = "synthetic-version"

    def __init__(self) -> None:
        self.calls = 0

    def verify(self) -> None:
        pass

    def classify(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> dict[str, Any]:
        assert schema["properties"]["intent"]["enum"] == LABELS
        assert "true_intent" not in json.dumps(messages)
        self.calls += 1
        return response('{"intent":"synthetic_0"}')


def fake_retriever() -> Any:
    return SimpleNamespace(
        binding={"baseline": "synthetic"},
        retrieve=lambda text, k: {
            "matches": [
                {"id": "fitting-1", "text": "synthetic fit", "intent": LABELS[0], "similarity": 0.9}
            ],
            "descriptions": {LABELS[0]: "authored description"},
        },
    )


def test_graph_full_flow_cache_gold_isolation_and_compute(tmp_path: Path) -> None:
    pytest.importorskip("langgraph")
    pytest.importorskip("filelock")
    baseline = SimpleNamespace(
        labels=LABELS, top_k=lambda text, k: [SimpleNamespace(intent=LABELS[0])]
    )
    provider = FakeProvider()
    graph = build_graph(
        LocalConfig(),
        LABELS,
        baseline,
        fake_retriever(),
        provider,
        ResponseCache(tmp_path / "cache"),
    )
    frame = pd.DataFrame({"text": ["private synthetic request"] * 2, "intent": LABELS[:2]})
    rows, seconds = execute(graph, frame, tmp_path / "ledger.json")
    assert provider.calls == 1 and [r["cache_hit"] for r in rows] == [False, True]
    assert all(r["needs_review"] and r["agreement"] and r["route"] == "human_review" for r in rows)
    assert "private synthetic request" not in (tmp_path / "ledger.json").read_text()
    compute = summarize_compute(rows, seconds)
    assert compute["live"]["tokens"]["eval_count"]["known_total"] == 5
    assert compute["cache_hits"]["tokens"]["eval_count"]["known_total"] == 5
    assert compute["api_cost_usd"] == 0
    assert extrapolation(compute)["final_test_3080_seconds"][0] > 0
    gate = gate_diagnostics(rows, frame.intent.tolist())
    assert gate["agreement_only"]["accuracy"] == 0.5
    assert gate["operational_accepted_count"] == 0 and gate["tau"] is None
    with pytest.raises(ValueError, match="ledger"):
        execute(graph, frame, tmp_path / "ledger.json")
    baseline.labels = list(reversed(LABELS))
    with pytest.raises(ValueError, match="mapping"):
        graph.invoke({"text": "synthetic"})
    write_report(
        tmp_path / "report",
        {
            "status": "SYNTHETIC",
            "comparison": {
                "llm": quality(frame.intent.tolist(), [r["intent"] for r in rows], LABELS)
            },
            "compute": compute,
        },
    )
    assert "Weighted F1" in (tmp_path / "report.md").read_text()


def test_ollama_transport_options_identity_failures_and_fallback() -> None:
    import httpx

    seen = []
    digest = MODEL_DIGEST

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": MODEL_TAG, "digest": digest}]})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "synthetic"})
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": MODEL_TAG,
                "message": {"content": "{}"},
                "done": True,
                "done_reason": "stop",
                "eval_count": 2,
            },
        )

    provider = OllamaProvider(LocalConfig())
    provider.client.close()
    provider.client = httpx.Client(
        base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handler)
    )
    provider.verify()
    assert provider.version == "synthetic"
    assert provider.classify([], output_schema(LABELS))["eval_count"] == 2
    assert seen[0]["options"] == {"temperature": 0, "seed": 42, "num_predict": 48, "num_ctx": 4096}
    provider.config = replace(provider.config, mechanism="strict_parser")
    provider.classify([], output_schema(LABELS))
    assert "format" not in seen[-1]
    digest = "wrong"
    with pytest.raises(ValueError, match="digest"):
        provider.verify()
    provider.client.close()
    provider.client = httpx.Client(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="private provider failure")
        ),
    )
    assert provider.classify([], {})["failure"] == "local_provider_failure"
    provider.client.close()


def test_local_encoder_chroma_and_binding_are_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {}

    def encoder(*args: Any, **kwargs: Any) -> Any:
        options.update(kwargs)
        return SimpleNamespace(encode=lambda texts, **kw: np.ones((len(texts), 2)))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=encoder)
    )
    local = LocalEncoder(LocalConfig(), tmp_path)
    assert options["local_files_only"] and options["trust_remote_code"] is False
    assert options["revision"] == LocalConfig().embedding_revision
    assert local.encode(["synthetic"]).shape == (1, 2)
    descriptions = tmp_path / "descriptions.json"
    atomic_json(descriptions, {name: "synthetic description" for name in LABELS})
    binding = index_binding(
        LocalConfig(),
        LABELS,
        {
            "baseline_sha256": "baseline",
            "manifest_sha256": "manifest",
            "dataset_checksums": {"train": "training"},
        },
        descriptions,
        [1, 2],
    )
    assert binding["ordered_labels_sha256"] != fingerprint(list(reversed(LABELS)))
    atomic_json(tmp_path / "binding.json", binding)
    examples = SimpleNamespace(
        count=lambda: 8002,
        query=lambda **kw: {
            "ids": [["fit-1"]],
            "documents": [["synthetic fit"]],
            "metadatas": [[{"intent": LABELS[0]}]],
            "distances": [[0.1]],
        },
    )
    definitions = SimpleNamespace(
        count=lambda: 77,
        get=lambda **kw: {
            "ids": [LABELS[0]],
            "documents": ["synthetic description"],
        },
    )
    client = SimpleNamespace(
        get_collection=lambda name, **kw: examples if name == "fitting_examples" else definitions
    )
    monkeypatch.setattr("intentops.llm_path.chroma_client", lambda path: client)
    retriever = ChromaRetriever(tmp_path, local, binding)
    result = retriever.retrieve("synthetic", 5)
    assert result["matches"][0]["similarity"] == pytest.approx(0.9)
    assert result["descriptions"][LABELS[0]] == "synthetic description"
    with pytest.raises(ValueError, match="binding"):
        ChromaRetriever(tmp_path, local, {})
    examples.count = lambda: 1
    with pytest.raises(ValueError, match="incomplete"):
        ChromaRetriever(tmp_path, local, binding)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    assert load_config(config_path) == LocalConfig()


def test_unknown_usage_and_empty_acceptance_are_not_fabricated() -> None:
    record = {
        "cache_hit": False,
        "agreement": False,
        "intent": None,
        "nearest_similarity": 0.5,
        **{f"{key}_seconds": 1.0 for key in ("retrieval", "baseline", "llm", "end_to_end")},
        "prompt_eval_count": None,
        "eval_count": None,
        "prompt_eval_cached_count": None,
    }
    compute = summarize_compute([record], 1)
    assert compute["live"]["tokens"]["eval_count"]["unknown_rows"] == 1
    assert compute["cache_hits"]["latency"]["llm"]["mean_seconds"] is None
    assert gate_diagnostics([record], [LABELS[0]])["agreement_only"]["wilson_95"] is None
    assert extrapolation(summarize_compute([], 0))["validation_2001_seconds"] is None


def test_cli_refuses_official_test_before_opening_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_llm.py", "--split", "test"])

    def forbid(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("No dataset may be opened for an unsupported split.")

    monkeypatch.setattr(pd, "read_parquet", forbid)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/evaluate_llm.py"), run_name="__main__"
        )
    assert exit_info.value.code == 2


def test_pilot_requires_matching_smoke_gate(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_llm.py"))
    require_smoke = namespace["require_smoke"]
    path = tmp_path / "smoke.json"
    smoke = {
        "smoke_gate_passed": True,
        "rows": 20,
        "compute": {"live": {"latency": {"end_to_end": {"examples": 20, "mean_seconds": 1}}}},
        "configuration": {"config": 1},
        "index_binding": {"baseline": "synthetic"},
    }
    atomic_json(path, smoke)
    require_smoke(path, smoke["configuration"], smoke["index_binding"])
    with pytest.raises(ValueError, match="smoke"):
        require_smoke(path, {"config": 2}, smoke["index_binding"])
    smoke["compute"]["live"]["latency"]["end_to_end"]["mean_seconds"] = 11
    atomic_json(path, smoke)
    with pytest.raises(ValueError, match="smoke"):
        require_smoke(path, smoke["configuration"], smoke["index_binding"])


@pytest.mark.parametrize(
    "intent,baseline,nearest,similarity,tau,review",
    [
        ("a", "a", "a", 0.9, 0.8, False),
        ("a", "a", "a", 0.8, 0.8, True),
        ("a", "a", "a", 0.9, None, True),
        (None, "a", "a", 0.9, 0.8, True),
        ("a", "b", "a", 0.9, 0.8, True),
        ("a", "a", "b", 0.9, 0.8, True),
    ],
)
def test_gate_requires_all_agreements_and_strict_similarity(
    intent: str | None,
    baseline: str,
    nearest: str,
    similarity: float,
    tau: float | None,
    review: bool,
) -> None:
    assert routing_gate(intent, baseline, nearest, similarity, tau)["needs_review"] is review


def test_cache_concurrent_requests_resolve_once(tmp_path: Path) -> None:
    pytest.importorskip("filelock")
    from concurrent.futures import ThreadPoolExecutor

    cache = ResponseCache(tmp_path)
    calls = []

    def create() -> dict[str, Any]:
        calls.append(1)
        return {"intent": LABELS[0]}

    key = fingerprint("synthetic concurrency")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: cache.resolve(key, create), range(4)))
    assert len(calls) == 1 and sum(not hit for _, hit in results) == 1


def test_chroma_local_persistence_uses_explicit_synthetic_vectors(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from intentops.llm_path import chroma_client

    client = chroma_client(tmp_path)
    collection = client.create_collection(
        "synthetic_examples", embedding_function=None, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["a", "b"],
        documents=["synthetic A", "synthetic B"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    found = collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)
    assert found["ids"] == [["a"]]
    assert (
        chroma_client(tmp_path)
        .get_collection("synthetic_examples", embedding_function=None)
        .count()
        == 2
    )


def test_ledger_and_full_report_without_optional_graph_runtime(tmp_path: Path) -> None:
    class FakeCompiledGraph:
        def invoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
            assert set(payload) == {"text"} and config["recursion_limit"] == 12
            failed = payload["text"] == "synthetic failed request"
            return {
                "retrieved": {
                    "matches": [{"id": "fitting-1", "intent": LABELS[0], "similarity": 0.9}]
                },
                "response": response("{}", failure="synthetic_failure")
                if failed
                else response("{}"),
                "intent": None if failed else LABELS[0],
                "baseline_label": LABELS[0],
                "agreement": not failed,
                "needs_review": True,
                "route": "human_review",
                "cache_hit": failed,
                "cache_key": fingerprint(payload),
                "retrieval_seconds": 0.1,
                "baseline_seconds": 0.01,
                "llm_seconds": 0.2,
            }

    frame = pd.DataFrame(
        {"text": ["synthetic private request", "synthetic failed request"], "intent": LABELS[:2]},
        index=[10, 20],
    )
    ledger = tmp_path / "compute.json"
    rows, elapsed = execute(FakeCompiledGraph(), frame, ledger)
    assert [row["row_id"] for row in rows] == [10, 20]
    assert all(len(row["row_sha256"]) == 64 for row in rows)
    assert "synthetic private request" not in ledger.read_text()
    assert rows[1]["intent"] is None and rows[1]["failure"] == "synthetic_failure"
    compute = summarize_compute(rows, elapsed)
    assert compute["live"]["latency"]["retrieval"]["mean_seconds"] == 0.1
    assert compute["cache_hits"]["rows"] == 1
    with pytest.raises(ValueError, match="ledger"):
        execute(FakeCompiledGraph(), frame, ledger)
    from dataclasses import asdict

    report = {
        "status": "SYNTHETIC_PILOT",
        "rows": 2,
        "represented_classes": 2,
        "configuration": asdict(LocalConfig()),
        "compute": compute,
        "comparison": {
            "llm_rag": quality(frame.intent.tolist(), [row["intent"] for row in rows], LABELS)
        },
        "gate": gate_diagnostics(rows, frame.intent.tolist()),
        "time_extrapolation": extrapolation(compute),
        "invalid_output_count": 1,
        "failure_count": 1,
        "limitations": ["Synthetic evidence only"],
    }
    write_report(tmp_path / "pilot", report)
    exported = json.loads((tmp_path / "pilot.json").read_text())
    assert exported["comparison"]["llm_rag"]["accuracy"] == 0.5
    markdown = (tmp_path / "pilot.md").read_text()
    assert "Macro F1" in markdown and "Weighted F1" in markdown and "τ PENDING" in markdown
    assert "Unknown rows" in markdown and "cache_hits" in markdown and "Time planning" in markdown
    empty = {
        "status": "SYNTHETIC_EMPTY",
        "compute": summarize_compute([], 0),
        "time_extrapolation": extrapolation(summarize_compute([], 0)),
    }
    write_report(tmp_path / "empty", empty)
    assert "PENDING" in (tmp_path / "empty.md").read_text()
