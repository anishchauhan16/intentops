# Final publication review

Status: local candidate only. No remote repository was created or pushed and no Render
service was created. Public publication and deployment await the owner's explicit approval.

## Candidate

- Name: `intentops`
- Title: **IntentOps — ML-Powered Intent Classification & Human-Review Routing**
- Proposed description: “77-class banking intent routing with FastAPI, human-review
  gating, reproducible evaluation, and a Docker demo.”
- Publication/build context: the allowlisted `demo-release/` directory.
- Proposed source license: MIT; separate BANKING77 attribution in `NOTICE.md`.
- Baseline: `baseline-c8f5bc42f4b0`, SHA-256 recorded in artifact metadata and release.json.
- Default deployment: one Render Free Docker web service, no paid resources, no secrets,
  no persistent disk, semantic mode disabled, PORT=10000, readiness `/health`, auto-deploy off.

## Security review performed on 2026-10-21

- The root and prepared bundle had no Git metadata or history at the start of review.
  There was no existing history to scan or rewrite.
- Scanned release and non-environment workspace text files for common token/private-key
  patterns, credential assignments, embedded URL credentials and absolute local user/temp
  paths. The scan found no matches. This is a targeted scan, not a guarantee that arbitrary
  secrets can be detected automatically.
- `.env` files, caches, raw data, virtual environments and credentials are excluded from
  the release allowlist. The Docker context includes runtime source, pins, licenses and
  the two frozen baseline files only.
- `pip-audit` examined all 27 pinned runtime packages with dependency resolution disabled
  because their full closure is pinned. It reported **no known vulnerabilities** from
  its public advisory service at audit time. Re-audit before future releases.
- Confirmed debug mode is not enabled, startup errors use fixed messages, metadata exposes
  safe selected fields, inference logs omit request text, and the launcher disables URL
  access logs. HTML input is escaped and demo responses have CSP/no-store policies.
- Frozen model deserialization remains a trusted-artifact operation. A checksum detects
  mismatched bundles; it is not publisher authentication. The model contains vocabulary
  derived from public BANKING77 data, not private customer requests.

## Verified and pending checks

- Ruff, formatter and mypy pass; 71 offline tests pass with 98.74% branch-aware coverage.
- Existing artifact checksum, size, metrics and routing counts match the saved reports.
- Re-evaluation of the unchanged model on the official test Parquet exactly reproduced
  all saved metrics and confusion-matrix entries, including 1,383 correct predictions
  among 1,397 accepted requests at the unchanged 0.70 threshold. No model retraining or
  threshold selection was performed during this final review.
- A fresh non-editable installation of the publication candidate passed its dependency
  check, Ruff, mypy and all 71 offline tests. The workflow uses a read-only GitHub token
  and a 15-minute job timeout.
- Live **local** HTTP verification passed health, metadata, model version, PIN acceptance,
  card/ambiguous review, empty-input errors, oversized batches, demo responses and assets.
- The original 0.85/1.43 ms p50/p95 numbers are **in-process TestClient round-trip**
  measurements, not pure model latency or public network latency.
- Follow-up verification on 2026-10-22 built Linux amd64 and arm64 images. The native
  arm64 image passed health, artifact loading, functional HTTP checks, dependency checks,
  and UID 10001 verification with a read-only root, 512 MiB and 0.5 CPU. One post-smoke
  memory sample was 137.3 MiB. Disposable verification containers were removed.
- The amd64 runtime crashed during sklearn import under local QEMU, including without
  resource constraints. Native Ubuntu amd64 CI and hosted inference must pass before
  asserting target-platform verification. No model/dependency change hid this failure.
- GitHub CLI authentication is verified as `anishchauhan16`; an authenticated repository
  lookup found no existing `anishchauhan16/intentops` repository. Creation, public push,
  hosted CI and Render deployment remain pending approval. No hosted checks are reported
  as passing.
- README screenshots show the actual running local application with a fictional PIN
  request. The visible single-request timing is not a benchmark.
- Record the real repository/endpoint URLs and deployment verification only after they
  exist. Permission cleanup remains the final handoff step after deployment work.
