# Demo-readiness audit

Audit date: 2026-10-21. Scope: current source, synthetic tests, frozen baseline,
raw-data manifest and saved evaluation/benchmark outputs.

## Findings and changes

| Finding | Resolution |
| --- | --- |
| Existing Pipeline and deterministic 80/20 stratified split respected train/validation/test boundaries | Retained the model and configuration; verified split sizes and re-evaluated the frozen artifact without tuning |
| API worked but recruiters needed to compose JSON | Added an accessible server-rendered form with three examples, review state, classifier alternatives and model metadata |
| Unbounded raw HTTP body parsing preceded schema validation | Added a 1 MiB streamed-body limit, retaining 2,000-character / 32-item limits |
| HTML submission introduces a reflection surface | Escaped dynamic content, restricted CSP, no caching or raw-text persistence; regression-tested script markup |
| Docker dependencies could drift away from sklearn serialization version | Captured the installed runtime dependency closure; added explicit sklearn compatibility validation |
| Fixed container port and access logging were poor hosting defaults | Added PORT-aware, single-worker launcher, bounded concurrency, disabled URL access logs, bounded BLAS threads and PORT-aware health probe |
| Frozen artifacts and aggregate reports were absent from a fresh Git checkout | Added an allowlisted, local-only release packager and a tracked text-free summary; preserved ignored raw data and detailed errors |
| Existing container and workflow needed a repeatable handoff | Added concrete Render configuration and approval-gated manual deployment/verification instructions |

## Evidence and limits

- Original baseline: `baseline-c8f5bc42f4b0`, 8,002 fitting rows, 2,001 validation rows,
  3,080 official test rows. Fitting uses one sklearn Pipeline; inference never refits it.
- Original raw Parquet checksums were verified against the manifest. Re-evaluation of
  the unchanged frozen model exactly reproduced the saved metrics and confusion matrix.
- Exact duplicate text counts were zero within development/test, and exact text overlap
  was zero between fitting/validation, fitting/test and validation/test. This does not
  rule out near-duplicates or semantically overlapping queries.
- A fresh Python 3.11 environment installed the release as a built, non-editable package
  with pinned runtime dependencies and dev tools, without data/semantic extras. Ruff,
  format checks, mypy and all **71 tests** passed with **98.74% branch-aware coverage**.
  The same suite passes in the working environment. An upstream TestClient/httpx
  deprecation warning remains; it does not fail the suite.
- A real HTTP server using that isolated installation passed readiness, metadata,
  prediction, batch, invalid-input, body-limit, HTML/CSS, accepted/review demo and security
  header checks. The launcher logs contained neither example text nor request URLs.
- Fresh-environment training from the verified local train Parquet followed by
  validation/test evaluation exactly reproduced every saved baseline metric and
  confusion-matrix cell. This used no network or threshold tuning. The reserialized
  artifact had a new checksum-derived version; the original release artifact was retained.
- The browser form and a prediction/review result were inspected in Chrome. Local
  evidence is separate from cloud/Linux verification.
- [Aggregate evidence](../reports/summary.json) includes the model checksum, data hashes,
  quality metrics, routing counts, training runtime and measured benchmark scope.
- The optional semantic encoder remains local-cache-only at inference. Its vote is
  separate from classifier probability. The release intentionally contains no semantic
  weights or claimed hybrid quality gains.
- Unit tests block sockets and use synthetic fixtures. Semantic downloads are mocked.
- Follow-up on 2026-10-22 installed Docker Desktop and verified a native Linux arm64
  container at 512 MiB / 0.5 CPU with a read-only root, UID 10001 and healthy readiness.
  All service smoke checks passed; one post-smoke memory sample was 137.3 MiB.
- The amd64 image crashes during sklearn import under local QEMU on this Apple Silicon
  host, even without resource constraints. Follow-up native Ubuntu amd64
  [CI run](https://github.com/anishchauhan16/intentops/actions/runs/39729975061) passed all
  71 tests (98.74% branch-aware coverage), Ruff, mypy, Docker build and constrained
  container readiness/inference. No dependency/model change was required.
- After owner approval, published [the repository](https://github.com/anishchauhan16/intentops)
  and deployed commit `f1daf7629be9cabe9c496d7e1368c31faba02098` to
  [Render Free](https://intentops-demo.onrender.com/) on 2026-10-22. The external
  verification script passed health, metadata/model identity, accepted/review predictions,
  input/batch validation, demo HTML/CSS/docs and form decisions. The owner confirmed
  Auto-Deploy Off was saved. No paid compute was created.
- Hosted cold-start distribution, peak memory, concurrency and load performance remain
  unmeasured. In-process benchmark numbers must not be presented as hosted latency.
- `requirements-runtime.txt` pins Python dependencies; the Python base tag is pinned to
  a patch release, not an immutable OCI digest. Record an image digest after the approved
  build and review security updates periodically.
- Confidence is uncalibrated, the model is closed-set and English banking only. Review
  routing is not a real external queue. The public demo is intended for fictional input.
