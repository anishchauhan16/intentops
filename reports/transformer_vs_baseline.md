# Transformer versus frozen baseline

Status: **MEASURED_TEST_ONLY**.

All null fields mean **PENDING**, not zero.
Official test results from the single frozen final run.

| Metric | Baseline | Transformer |
| --- | ---: | ---: |
| accuracy | 0.876299 | 0.889610 |
| macro_f1 | 0.877201 | 0.888488 |
| weighted_f1 | 0.877201 | 0.888488 |
| top_3_accuracy | 0.961688 | 0.970779 |
| coverage_at_0_70 | 0.453571 | 0.737662 |
| accepted_accuracy_at_0_70 | 0.989979 | 0.976232 |
| artifact_bytes | 5781920 | 269018787 |
| training_seconds | PENDING | 356.657884 |
| CPU batch 1 mean_ms_per_example | 0.190485 | 5.812472 |
| CPU batch 1 p95_ms_per_example | 0.204077 | 6.055435 |
| CPU batch 16 mean_ms_per_example | 0.015870 | 3.074837 |
| CPU batch 16 p95_ms_per_example | 0.016190 | 4.788594 |
| Training hardware | PENDING | cpu, arm64, 2 CPU threads |

Baseline training duration/hardware: **PENDING**; no baseline retraining.

Latency uses the same CPU, examples, threads, batches and warmups for both models.
Batch p95 is amortized batch time per example, not per-request network tail latency.

See `run_evidence` in the JSON for actual evaluation observations, hardware, training time and shared CPU timings.
Smoke observations are pipeline checks, not transformer benchmark results.
