# Phase 2 local experiment

Status: **MEASURED_VALIDATION_PILOT_TAU_PENDING**.

API cost: **$0.00**. Official test split was not opened. Null means PENDING.

| Path | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| baseline | 0.848000 | 0.831839 | 0.836349 |
| transformer | 0.884000 | 0.875448 | 0.881952 |
| llm_rag | 0.908000 | 0.897722 | 0.901546 |

Model: `qwen2.5:7b-instruct-q4_K_M`; manifest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
Output mechanism: `json_schema`; temperature 0, seed 42, cap 48 tokens.
Split: validation; rows: 250; represented intents: 77.
Invalid outputs: 0; provider failures: 0.

| Calls | Stage | N | Mean seconds | p95 seconds |
| --- | --- | ---: | ---: | ---: |
| live | retrieval | 230 | 0.110151 | 0.317312 |
| live | baseline | 230 | 0.019604 | 0.053784 |
| live | llm | 230 | 2.270330 | 3.324585 |
| live | end_to_end | 230 | 2.452225 | 3.598826 |
| cache_hits | retrieval | 20 | 0.157389 | 0.768725 |
| cache_hits | baseline | 20 | 0.023537 | 0.060264 |
| cache_hits | llm | 20 | 0.009710 | 0.014949 |
| cache_hits | end_to_end | 20 | 0.247582 | 0.918088 |

sequential local batch=1; graph includes identity verification, retrieval, baseline, LLM HTTP and cache I/O; initialization excluded; first live load is included; ledger writes included only in total wall time; cache-hit token totals are historical

Graph-loop wall time including ledger writes: 575.887 s.

| Calls | Token field | Known total | Unknown rows |
| --- | --- | ---: | ---: |
| live | prompt_eval_count | 313838 | 0 |
| live | eval_count | 2302 | 0 |
| live | prompt_eval_cached_count | 235087 | 0 |
| cache_hits | prompt_eval_count | 27501 | 0 |
| cache_hits | eval_count | 225 | 0 |
| cache_hits | prompt_eval_cached_count | 19418 | 0 |

## Gate diagnostics — τ PENDING

Operational coverage is 0; all rows route to review. Accepted accuracy is PENDING.
Agreement-only candidate support: 201; coverage: 80.4000%; accuracy: 0.9800995024875622; 95% Wilson interval: [0.9499575439096812, 0.9922345353262929].
These candidates do not establish the 99% target or select a similarity threshold.

## Time planning

live mean/p95; planning range, not a confidence interval; no cache

- validation_2001_seconds: 81.78–120.02 minutes.
- final_test_3080_seconds: 125.88–184.74 minutes.

Neither projected run was performed. Both require later approval.

## Limitations

- 250 validation rows only; no tau selection or full validation.
- Agreement-only accuracy is diagnostic; no tuned acceptance threshold.
- 99% is a target; inspect support and Wilson interval, not assurance.
- Mean/p95 extrapolation is a planning range, not an interval on total runtime.
- Local timings differ from hosted APIs and the Phase 1 CPU protocol.
- Models were loaded, never retrained. Official test split was never opened.

Machine-readable evidence (including timing, token accounting, row IDs/hashes,
configuration and limitations) is in [phase2_pilot.json](phase2_pilot.json).
