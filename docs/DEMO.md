# Recruiter walkthrough (60 seconds)

Start locally: `source .venv/bin/activate && python -m intentops.serve` and open
`http://localhost:8000/`. A future public URL must be verified before it is shared.

| Time | Action and narration |
| --- | --- |
| 0–10s | “IntentOps routes banking support requests across 77 intents and flags uncertainty for human review. It runs a classical model locally on CPU.” |
| 10–25s | Click **Change a PIN**, then **Predict intent**. The frozen baseline predicts `change_pin` at about 96.5% confidence and automatically accepts it at the 70% gate. Point to the alternatives. |
| 25–40s | Click **A card that hasn’t arrived**, then predict. It suggests `card_arrival` at about 57.9% and flags human review. “A prediction is still useful for triage even when automation should abstain.” |
| 40–50s | Click **An unclear request**, then predict. “This closed-set model still suggests an intent for vague text; confidence helps route it to review, but does not prove out-of-domain detection.” |
| 50–60s | Open **API docs** or `/metadata`. “The official 3,080-row test split measured 87.63% accuracy and 96.17% top-3. At the fixed gate, accepted accuracy was 99.00% at 45.36% coverage. The repo includes offline tests, CI and a non-root Docker image.” |

Confidence examples above were checked against `baseline-c8f5bc42f4b0`. Treat them as
examples from that frozen model, not a general accuracy promise. The browser displays
routing time, excluding network transport and rendering. Review means a boolean flag;
no real support ticket or external queue is created.

If asked about semantic fallback: it is optional, lazy-loaded, and covered with mocked
encoders. No real hybrid quality improvement has been measured in the shipped demo.

## Resume bullets

- **SWE:** Built a typed FastAPI support-routing service with a server-rendered demo,
  bounded request validation, model-readiness checks, structured privacy-conscious logs,
  71 offline regression tests, CI, and a non-root Docker deployment configuration.
- **AI Engineer:** Implemented a 77-intent banking router with TF-IDF/logistic regression,
  configurable human-review gating and optional local embedding retrieval; measured
  99.00% accepted accuracy at 45.36% coverage on the official BANKING77 test split.
- **ML Engineer:** Built a reproducible BANKING77 training/evaluation pipeline with
  stratified development splitting, checksummed artifacts, confusion/error reports and
  threshold analysis; achieved 87.63% accuracy, 0.8772 macro F1 and 96.17% top-3 accuracy
  on 3,080 held-out examples.

These bullets describe implemented and measured work. They intentionally make no claim
of public deployment, production traffic, semantic quality gains or business impact.
