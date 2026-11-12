# Phase 2 local experiment

Status: **SMOKE_GATE_PASSED**.

API cost: **$0.00**. Official test split was not opened. Null means PENDING.

Model: `qwen2.5:7b-instruct-q4_K_M`; manifest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
Output mechanism: `json_schema`; temperature 0, seed 42, cap 48 tokens.
Split: validation; rows: 20; represented intents: 20.
Invalid outputs: 0; provider failures: 0.

| Calls | Stage | N | Mean seconds | p95 seconds |
| --- | --- | ---: | ---: | ---: |
| live | retrieval | 20 | 0.034001 | 0.137658 |
| live | baseline | 20 | 0.001827 | 0.005122 |
| live | llm | 20 | 1.469887 | 1.721599 |
| live | end_to_end | 20 | 1.513819 | 1.761693 |
| cache_hits | retrieval | 0 | PENDING | PENDING |
| cache_hits | baseline | 0 | PENDING | PENDING |
| cache_hits | llm | 0 | PENDING | PENDING |
| cache_hits | end_to_end | 0 | PENDING | PENDING |

sequential local batch=1; graph includes identity verification, retrieval, baseline, LLM HTTP and cache I/O; initialization excluded; first live load is included; ledger writes included only in total wall time; cache-hit token totals are historical

Graph-loop wall time including ledger writes: 30.321 s.

| Calls | Token field | Known total | Unknown rows |
| --- | --- | ---: | ---: |
| live | prompt_eval_count | 27501 | 0 |
| live | eval_count | 225 | 0 |
| live | prompt_eval_cached_count | 19418 | 0 |
| cache_hits | prompt_eval_count | 0 | 0 |
| cache_hits | eval_count | 0 | 0 |
| cache_hits | prompt_eval_cached_count | 0 | 0 |

Machine-readable evidence (including timing, token accounting, row IDs/hashes,
configuration and limitations) is in [phase2_smoke.json](phase2_smoke.json).
