# Phase 1: optional transformer training and evaluation

This is an independent PyTorch training path for the existing 77-class task. It does
not replace the FastAPI baseline, alter API contracts, or change Render's default model.
`requirements-runtime.txt`, Dockerfile and Render configuration remain unchanged.
The default is **distilbert-base-uncased**, not a substituted smaller model.

## Data discipline and canonical labels

`development_data()` calls the existing `read_data()` and `split_development()`.
The unchanged 10,003 official training rows are split with stratification, seed 42,
and validation fraction 0.20: 8,002 fitting rows and 2,001 validation rows. Validation
macro F1 selects the best epoch and controls early stopping (patience 1). No fitting,
vocabulary construction, early stopping or threshold selection uses official test data.
The pretrained tokenizer is retained; no tokenizer is fitted on validation/test text.

The baseline metadata's ordered `label_names` is canonical. Dataset manifest integer
ordering differs, so integer labels are ignored: intent **names** are translated into
the baseline order. The code asserts the same 77 distinct names as the manifest, then
exact transformer configuration/checkpoint ordering against baseline metadata when
building or loading. Synthetic tests deliberately reverse manifest order.

The review gate is fixed at 0.70 for both models; softmax probabilities are not assumed
calibrated. All examples, including reviewed ones, count toward overall metrics.

## Environment and installation

Work in the Git checkout (`demo-release/` in the original workspace), on
`feat/transformer-path`. Do not initialize Git at its parent workspace.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-runtime.txt -e ".[dev,data,transformer]"
# Versions actually used for this experiment:
python -m pip install torch==2.15.0 transformers==4.57.6 \
  tokenizers==0.22.2 safetensors==0.8.0
export HF_HOME="$PWD/.hf-cache"
export TOKENIZERS_PARALLELISM=false
```

Use a PyTorch installation compatible with your GPU's CUDA runtime if using CUDA;
see the [official installation selector](https://pytorch.org/get-started/locally/).
The implementation needs no Trainer, Accelerate, hosted inference API or paid service.
Weights are downloaded only by the explicit training command (unless `--offline`).
Checkpoint loading/evaluation uses local files only and disables remote model code.

In a fresh checkout, fetch the pinned public dataset:

```bash
python scripts/fetch_data.py --output-dir data/raw
```

In the original workspace, verified data already exists one directory above this Git
checkout. Use `--data-dir ../data/raw` instead; this session did not refetch the dataset.
Training reads only `train.parquet`; the test checksum in the manifest is provenance,
not a read of `test.parquet`.

## CPU smoke test

```bash
python scripts/train_transformer.py --data-dir data/raw --smoke
python scripts/evaluate_transformer.py --data-dir data/raw \
  --artifact-dir artifacts/transformer-smoke \
  --output-dir reports/transformer/smoke \
  --comparison reports/transformer/smoke-comparison
```

Smoke mode explicitly uses CPU, one epoch, two training examples per class (154), one
validation example per class (77), length 32 and batch 8. It uses the same pretrained
default model; only offline unit tests use a tiny random transformer. Smoke checkpoints
cannot freeze configuration or access the official test set. Their comparison quality
fields remain PENDING and their observations are segregated in `run_evidence`.

The real smoke run in this session completed on Apple M5 / 16 GB, macOS arm64,
Python 3.11.17, CPU with two PyTorch threads, seed 42 and CUDA unavailable. Training
took **3.549420 seconds**, excluding **6.974955 seconds** of model initialization and
download. These are pipeline smoke measurements, not benchmark quality results.

## Full training, validation, configuration freeze, final test

The pretrained revision resolved in this session is
`12040accade4e8a0f71eabdb258fecc2e7e948be`. Pin it when reproducing the run:

```bash
python scripts/train_transformer.py --data-dir data/raw \
  --model-name distilbert-base-uncased \
  --revision 12040accade4e8a0f71eabdb258fecc2e7e948be \
  --device cuda --artifact-dir artifacts/transformer
```

Use `--device cpu` without a CUDA GPU. CPU is supported; `auto` chooses CUDA if
available and otherwise CPU. MPS is not used. Output directories must not exist:
choose a new directory for another training experiment instead of overwriting a model.
Default settings are three epochs, learning rate 2e-5, AdamW weight decay 0.01,
batch 16, maximum length 128, gradient clipping 1.0 and seed 42. Python, NumPy,
PyTorch and shuffle generator seeds are set; deterministic algorithms are enabled.
Cross-version/hardware bitwise reproducibility is not guaranteed.

Validate and make all configuration decisions before test access. The default
evaluation split is **validation**, unlike the existing baseline evaluation CLI:

```bash
python scripts/evaluate_transformer.py --data-dir data/raw \
  --artifact-dir artifacts/transformer --split validation \
  --timing-examples 64 --batch-size 16 --warmup 3 --repeats 3 \
  --freeze reports/transformer/frozen.json
```

The freeze file binds checkpoint contents and metadata, baseline checksum, dataset
manifest, canonical labels, threshold and timing configuration. Stop here if you still
need to change anything. After decisions are final, run the following **once**:

```bash
python scripts/evaluate_transformer.py --data-dir data/raw \
  --artifact-dir artifacts/transformer --split test --final-test \
  --timing-examples 64 --batch-size 16 --warmup 3 --repeats 3 \
  --freeze-file reports/transformer/frozen.json
```

An exclusive attempt receipt is created in the checkpoint directory **before** test
access. A failed attempt also consumes the attempt: inspect its reason and seek review
before any recovery; never delete the receipt to repeatedly examine test scores.
The final command verifies the official test checksum, loads all 3,080 rows once and
evaluates both frozen models. It neither filters nor refits. Reusing the loaded texts
for the fixed timing protocol is part of that same final evaluation, not another
configuration-selection pass. Do not train/select another configuration based on the
final test results. The receipt prevents accidental reruns for this checkpoint; it is
an audit guard, not a security boundary against deliberate copying or tampering.

## Artifacts and reports

Checkpoints contain safetensors, tokenizer files, model config and metadata: ordered
labels, seed/configuration, split sizes, dataset checksums/source revisions, baseline
binding, resolved pretrained revision, package versions, hardware, selected epoch,
aggregate training history and measured training time. A SHA-256 manifest binds model,
configuration and tokenizer files. Loading verifies these and the canonical labels.
Only load trusted artifacts; checksums are not publisher authentication.

Training time includes optimizer steps, per-epoch validation and best-state restoration;
it excludes model initialization/download and checkpoint writing. Baseline training
time/hardware remain PENDING because they were not recorded; the baseline is not retrained.

Evaluation writes accuracy, macro/weighted F1, top-3 accuracy, a 77-label confusion
matrix, directed confusion pairs, misclassification CSVs and selective coverage/accuracy
at 0.70. Example exports stay in ignored `reports/transformer/`; console logs contain
only aggregate progress. The versioned comparison includes no raw support text:

- [Comparison table](../reports/transformer_vs_baseline.md)
- [Machine-readable evidence](../reports/transformer_vs_baseline.json)

## Shared CPU latency protocol

Both models are re-timed in the same process on the same machine and the same 64 texts,
sampled with seed 42 from the evaluated split. A SHA-256 of the ordered text list is
recorded and compared without exporting those texts. PyTorch and BLAS each use one CPU
thread. Models are loaded before timing. No concurrent training is used during the
final timing run.

Batch sizes are 1 and 16. Each model/size excludes three warmup batches, then processes
all 64 examples three times: 192 measured batch-1 calls and 12 measured batch-16 calls.
Timing includes TF-IDF/tokenization, model forward computation and conversion to
probabilities; it excludes startup, transport, HTTP validation/logging and rendering.
Dynamic padding and the transformer's saved truncation length apply normally.

Mean and p95 are computed over elapsed batch milliseconds divided by batch size.
With equal full batches, the mean is time per example. For batch 16, **p95 is amortized
batch time per example**, not an individual-request tail latency. Small samples, model
order, local thermal/load conditions and one run limit statistical confidence. These
are sequential in-process measurements, not network SLAs or concurrent-load results.
The historical 0.85/1.43 ms baseline TestClient p50/p95 used a different protocol and is
not included in this side-by-side comparison.

## Tests and runtime isolation

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest
```

All unit fixtures are synthetic. Transformer fixtures instantiate a randomly initialized
one-layer DistilBERT with a local nine-token vocabulary; tests download neither weights
nor data. They test canonical label equality, known metrics, checkpoint round trips,
checksum rejection, split disjointness, timing warmup/batch accounting and a synthetic
one-shot final evaluation. Without the optional extra, only the five transformer-dependent
tests skip; baseline tests still pass. CI adds an optional-extra test job with network
downloads disabled during tests, while the baseline job remains intact.

`.gitignore` excludes `artifacts/transformer*/`, local model cache, raw datasets,
per-example reports, virtual environments and common checkpoint extensions. No new
model checkpoints or datasets are committed. The two pre-existing baseline files remain
unchanged. Nothing in this phase deploys, pushes remotely or installs torch into Render.

## Limitations

This remains closed-set English banking classification. Softmax confidence is uncalibrated;
the shared 0.70 gate is not an assertion of equivalent calibration across architectures.
The tokenizer may truncate long requests. One seeded run cannot establish robustness,
fairness, statistical significance, or real support/business impact. A pretrained model
is substantially larger than the baseline and may increase latency despite better
quality. Optional semantic retrieval is excluded from this experiment.

## Session outcome

Completed on **2026-11-10**, using Apple M5 / 16 GB, CPU (CUDA unavailable),
Python 3.11.17, two training threads and one timing thread. All 8,002 fitting rows
and 2,001 validation rows were used; three epochs completed and validation selected
epoch 3. Measured optimization/validation time: **356.657884 seconds**.
Model version: `transformer-ee3848a11aaf`. The full initialization used the already
cached default pretrained revision; no hosted inference or paid resource was used.

The frozen final run evaluated all **3,080 official test examples once**. Accuracy was
**88.96%**, macro/weighted F1 **0.8885**, and top-3 accuracy
**97.08%**. The unchanged baseline reproduced **87.63%**
accuracy in this session. The transformer gains **1.33 percentage
points** overall, but accepted accuracy at 0.70 falls from **99.00%**
to **97.62%** as coverage grows from **45.36%** to
**73.77%**. This illustrates why the same numerical softmax gate is not
an equal-risk operating point across models. It was not retuned after test access.

The transformer bundle is approximately **46.5 times** the baseline
bundle size, and its measured batch-1 mean CPU time is approximately
**30.5 times** higher under this protocol. These measured costs and the modest
accuracy gain support keeping the smaller baseline in the live demo for now.

The separate smoke run also completed end to end. All **88 offline tests** pass,
including all 71 existing tests, with **98.62% branch-aware coverage**; Ruff, format
checks and mypy pass. The base environment without torch/transformers passes 83 tests
and skips only five dependency-specific tests. The new optional CI job is prepared;
no remote push or hosted run of this feature branch was performed.

There are no pending training/test runs for this recorded experiment. Baseline training
time/hardware remain PENDING by design. Future experiments need a new development plan;
do not rerun this checkpoint's official-test command or use its scores to tune settings.
The local checkpoint, frozen configuration, validation evidence and final-test receipt
are retained in ignored directories.
