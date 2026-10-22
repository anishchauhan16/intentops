# Demo deployment runbook

Status: prepared locally only. No cloud account, paid resource, registry upload or public
deployment has been created. Obtain the owner's approval before those actions.

## Hosting choice

Use one **Render Free Docker web service** for the first public demo. The saved baseline
is 5,697,572 bytes (5.43 MiB), with 8,903 TF-IDF features and 77 labels. It requires CPU
inference, no GPU, database, persistent disk, credentials or runtime network downloads.
A local macOS model-load process peaked at 196,853,760 bytes (187.7 MiB). This is an
observed local peak, not a Linux container memory limit or concurrent-load benchmark.

Render's free web instance offers 512 MB RAM and 0.1 CPU. That is a reasonable starting
point given the local footprint; verify actual container memory after deployment.
[Compute specifications](https://render.com/docs/compute-plans).

Free instances sleep after 15 idle minutes and can take about a minute to wake. There
are monthly usage quotas and an ephemeral filesystem. Preload the demo before an
interview. Quota exhaustion can suspend a free service; adding a payment method may
permit overage charges. [Free-service limits](https://render.com/docs/free).

For an always-on resume link, the 0.5 CPU / 512 MB tier is listed at **$7/month** as
checked on 2026-10-21; recheck before approving spend. [Pricing](https://render.com/pricing).
Hugging Face Docker Spaces are an alternative, but current documentation requires a
paid plan to create compute Spaces even though CPU Basic has no hourly charge.
[Spaces overview](https://huggingface.co/docs/hub/spaces-overview).

## Local preflight

From the repository root, with the supplied artifacts present:

```bash
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt -e ".[dev]"
make lint typecheck test
python -m intentops.serve
```

Open `http://localhost:8000/`; API docs are at `/docs`. Stop with Ctrl-C.
For a new checkout without artifacts, follow the README data/training commands first.
Keep `requirements-runtime.txt` aligned with the artifact's training environment.

```bash
docker build -t intentops:demo .
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --cap-drop ALL --security-opt no-new-privileges \
  --memory 512m --cpus 0.5 -e PORT=8000 -p 8000:8000 intentops:demo
```

Check `curl -fsS http://localhost:8000/health` and a browser prediction. Inspect
`docker stats --no-stream` under representative requests before assuming 512 MB is
sufficient. Docker must be installed and running. On 2026-10-22, native Linux arm64
container health/inference passed under these limits (137.3 MiB in one post-test
sample, not a peak/load measurement). The amd64 build succeeds, but local QEMU crashes
during sklearn import on Apple Silicon. Use the native architecture locally; require
Ubuntu amd64 CI and hosted smoke checks before claiming target deployment verification.

## Prepare a reviewable release without uploading

```bash
python scripts/package_demo.py
```

This creates ignored `demo-release/` with source, docs, tests, pinned runtime dependencies,
configuration, public aggregate metrics, and exactly the two frozen baseline artifacts.
It excludes raw data, detailed example reports, `.env`, credentials, virtual environments,
Git history and semantic weights. `release.json` records the model version and file list.
Existing destinations are rejected; choose `--output another-directory` for a new release.
Review the generated file list before publishing anything. The release's `.gitignore`
intentionally permits its frozen public-data baseline and metadata.

## Publish only after approval

The following steps upload source/artifacts and create a public endpoint. They have
not been executed. No secret or API token is required by the application.

1. Review `demo-release/release.json`. After approval, create a GitHub repository for
   this deployment bundle. A private source repository is sufficient for a public demo.
   From the bundle directory, initialize and push it using your normal Git credentials:

   ```bash
   cd demo-release
   git init -b main
   git add .
   git diff --cached --stat
   git commit -m "Prepare frozen IntentOps demo"
   git remote add origin <your-approved-repository-url>
   git push -u origin main
   ```

2. After approval for account creation if needed, open Render → **New → Web Service**.
   Connect the approved repository. Choose **Docker**, Dockerfile `./Dockerfile`,
   build context `.`, **Free** compute, and health-check path `/health`.
3. Set `PORT=10000`, `INTENTOPS_REVIEW_THRESHOLD=0.70`, and
   `INTENTOPS_SEMANTIC_ENABLED=false`. Leave the Docker start command unchanged;
   `python -m intentops.serve` reads `PORT`. Select **Auto-Deploy: Off** before creating
   the service. No database, disk, secret or paid resource is needed.
4. Confirm the selected free plan and approve the public deployment, then create the
   service. `render.yaml` contains the same configuration as a reference. Applying a
   Blueprint also creates a service on its initial sync; `autoDeployTrigger: off` only
   disables subsequent automatic deployments. Do not apply it without approval.
5. Wait for a successful build and `/health` readiness. Replace the URL below with the
   actual assigned endpoint and verify it externally:

   ```bash
   export DEMO_URL=https://your-assigned-service.onrender.com
   curl -fsS "$DEMO_URL/health"
   curl -fsS "$DEMO_URL/metadata"
   curl -fsS "$DEMO_URL/predict" -H 'Content-Type: application/json' \
     -d '{"text":"How do I change my PIN?"}'
   curl -sS -o /dev/null -w '%{http_code}\n' "$DEMO_URL/predict" \
     -H 'Content-Type: application/json' -d '{"text":""}'
   ```

   Expect health 200 with `model_loaded=true`, 77 metadata classes, a non-review
   `change_pin` prediction with the release model version, and 422 for empty text.
   Open `/`, submit all three examples, inspect `/docs`, and verify the browser has
   no failed static-file requests. Only then record the live URL on the resume/README.

   A repeatable functional check is included (uses fictional examples only):

   ```bash
   python scripts/verify_service.py "$DEMO_URL" \
     --expected-version baseline-c8f5bc42f4b0
   ```

   Use the version in the approved artifact metadata if the model changes. The script
   checks model identity, 77 classes, the fixed 0.70 gate, accepted/review predictions,
   invalid input, batch bounds, form responses and static assets. It does not benchmark
   service latency. The public release's CI runs this check against its Docker image.

Render supports Dockerfile deployments and configurable readiness paths.
[Docker setup](https://render.com/docs/docker),
[health checks](https://render.com/docs/health-checks),
[manual deployment settings](https://render.com/docs/deploys).

## Environment, secrets and scope

- `.env.example` contains only public defaults. Local execution reads process variables;
  it does not automatically read `.env`. Docker can accept `--env-file .env.example`.
- `INTENTOPS_ARTIFACT_DIR` defaults to `artifacts`; Docker sets `/app/artifacts`.
  The file path is not a secret. Baseline startup never downloads models or data.
- Run one process with BLAS/OpenMP threads limited to 1. The launcher bounds concurrent
  connections/tasks at 16; overload can return 503. This is not a per-user rate limiter.
- The 1 MiB transport cap and 2,000-character / 32-item schema limits apply to requests.
  `/health` fails with 503 when model initialization fails. A mismatched sklearn runtime
  is rejected before joblib loading.
- No customer text is logged by the application; the launcher disables URL access logs.
  Demo HTML has a restrictive content security policy and is not cached. Hosting access
  logs remain subject to platform policy. Use fictional examples only.
- The public demo has no authentication or persistent user data. Configure edge rate
  limits/authentication before accepting private traffic. Application tokens are not
  embedded in HTML or Docker layers. If future integrations require secrets, use the
  platform's secret settings, never source files, Docker build arguments, or Git.
- The image runs as a non-root user. The Docker build context is allowlisted. Updating
  runtime pins requires regression tests and renewed artifact/evaluation compatibility
  checks; pins do not eliminate the need for security updates.
- Roll back using the platform's prior deploy, or redeploy a reviewed prior release.
  Freeze the model, metadata, runtime pins and report summary together. Avoid build-time
  training, data downloads or mutable remote model URLs.
