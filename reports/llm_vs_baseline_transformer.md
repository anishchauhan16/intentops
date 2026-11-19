# Local LLM/RAG versus frozen baseline and transformer

Single final test evaluation; configuration frozen before test access. API cost $0.00.

## Validation — 2001 rows

| Path | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| baseline | 87.256372% | 0.874325 | 0.872686 |
| transformer | 89.905047% | 0.893909 | 0.899004 |
| llm_rag | 92.153923% | 0.923046 | 0.922085 |

Frozen τ: `0.8959192633628845`. Accepted: 1018 (1008 correct), review: 983; coverage: **50.8746%**; accepted accuracy: **99.0177%**; Wilson 95%: **98.2013%–99.4656%**.
Invalid outputs: 0; failures: 0.

| Calls | Stage | N | Mean seconds | p95 seconds |
| --- | --- | ---: | ---: | ---: |
| live | retrieval | 1751 | 0.020178031772566583 | 0.059424228966236115 |
| live | baseline | 1751 | 0.002627433103741679 | 0.009343916550278664 |
| live | llm | 1751 | 2.0618365346359675 | 3.1003201870480552 |
| live | end_to_end | 1751 | 2.0961453258043963 | 3.193001979147084 |
| cache_hits | retrieval | 250 | 0.004050958812236786 | 0.004745675052981823 |
| cache_hits | baseline | 250 | 0.0003360241698101163 | 0.00042469556210562577 |
| cache_hits | llm | 250 | 0.0004175428096204996 | 0.0005599500145763158 |
| cache_hits | end_to_end | 250 | 0.00872062447760254 | 0.011236587457824498 |

| Calls | Token field | Known total | Unknown rows |
| --- | --- | ---: | ---: |
| live | prompt_eval_count | 2393541 | 0 |
| live | eval_count | 17768 | 0 |
| live | prompt_eval_cached_count | 1788690 | 0 |
| cache_hits | prompt_eval_count | 341339 | 0 |
| cache_hits | eval_count | 2527 | 0 |
| cache_hits | prompt_eval_cached_count | 254505 | 0 |

Monotonic graph-loop elapsed (`perf_counter`): 3731.791 seconds. Overall monotonic elapsed including setup/comparisons: 3744.560 seconds.
sequential local batch=1; graph includes identity verification, retrieval, baseline, LLM HTTP and cache I/O; initialization excluded; first live load is included; ledger writes included only in total wall time; cache-hit token totals are historical

## Test — 3080 rows

| Path | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| baseline | 87.629870% | 0.877201 | 0.877201 |
| transformer | 88.961039% | 0.888488 | 0.888488 |
| llm_rag | 92.987013% | 0.929340 | 0.929340 |

Frozen τ: `0.8959192633628845`. Accepted: 1731 (1704 correct), review: 1349; coverage: **56.2013%**; accepted accuracy: **98.4402%**; Wilson 95%: **97.7401%–98.9258%**.
Invalid outputs: 0; failures: 0.

| Calls | Stage | N | Mean seconds | p95 seconds |
| --- | --- | ---: | ---: | ---: |
| live | retrieval | 3080 | 0.010090507675557966 | 0.020212118455674492 |
| live | baseline | 3080 | 0.0014330283632840622 | 0.0029373668250627784 |
| live | llm | 3080 | 1.4513244946855393 | 1.9599902919959273 |
| live | end_to_end | 3080 | 1.4688709177729953 | 1.9825640939641738 |
| cache_hits | retrieval | 0 | — | — |
| cache_hits | baseline | 0 | — | — |
| cache_hits | llm | 0 | — | — |
| cache_hits | end_to_end | 0 | — | — |

| Calls | Token field | Known total | Unknown rows |
| --- | --- | ---: | ---: |
| live | prompt_eval_count | 4190788 | 0 |
| live | eval_count | 30778 | 0 |
| live | prompt_eval_cached_count | 3153605 | 0 |
| cache_hits | prompt_eval_count | 0 | 0 |
| cache_hits | eval_count | 0 | 0 |
| cache_hits | prompt_eval_cached_count | 0 | 0 |

Monotonic graph-loop elapsed (`perf_counter`): 4648.667 seconds. Overall monotonic elapsed including setup/comparisons: 4661.695 seconds.
sequential local batch=1; graph includes identity verification, retrieval, baseline, LLM HTTP and cache I/O; initialization excluded; first live load is included; ledger writes included only in total wall time; cache-hit token totals are historical

Validation target: **MET** under the declared rule. **Test accepted accuracy is below 99%; the configuration was not retuned.**
The frozen baseline and transformer reproduce their published accuracies exactly.

Validation selects only τ. Test results do not select any model, prompt or threshold.
Wilson intervals describe observed support; they do not guarantee deployment accuracy.
Local timings differ from hosted API latency and Phase 1 CPU benchmarks.
No retraining, serving changes, deployment, push or merge occurred.

[Machine-readable evidence](llm_vs_baseline_transformer.json)

## Clock audit

The test receipt timestamps span **10,956.575 seconds (182.61 minutes)**, while
the monotonic graph-loop timer records **4,648.667 seconds (77.48 minutes)**.
The cause of this difference is unverified; suspension/clock-adjustment status was
not captured. Per-request and overall timer fields also use `perf_counter`. These
measurements must not be presented as interchangeable calendar elapsed time.
Validation has no recorded absolute start timestamp; its timer measurements are
reported without inventing a calendar duration. No inference was rerun.

The completed attempt receipt is preserved unchanged. Its export checksum binds
the original export archived in ignored `reports/llm-runs/final-test/export-at-completion.json`;
the public JSON adds this post-run clock audit without changing any result or configuration.
