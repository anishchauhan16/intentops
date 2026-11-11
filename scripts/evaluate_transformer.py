"""Evaluate validation by default; official test requires a frozen, one-shot final run."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from intentops.data import read_data, sha256_file
from intentops.model import Baseline
from intentops.transformer_evaluation import cpu_timing, evaluate_probabilities, write_comparison
from intentops.transformer_model import (
    TransformerModel,
    canonical_labels,
    dependencies,
    development_data,
    seed_everything,
)


def frozen_configuration(
    metadata: dict[str, Any], baseline_dir: Path, manifest: Path, timing: dict[str, Any]
) -> dict[str, Any]:
    return {
        "model_sha256": metadata["model_sha256"],
        "checkpoint_metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest(),
        "baseline_sha256": sha256_file(baseline_dir / "baseline.joblib"),
        "manifest_sha256": sha256_file(manifest),
        "threshold": 0.7,
        "timing_configuration": timing,
        "label_names": metadata["label_names"],
    }


def claim_final_test(
    artifact_dir: Path, freeze: Path, configuration: dict[str, Any], smoke: bool
) -> Path:
    if smoke:
        raise ValueError("Smoke checkpoints cannot access the official test set.")
    if json.loads(freeze.read_text()) != configuration:
        raise ValueError(
            "Frozen configuration does not match; validate and freeze before final test."
        )
    receipt = artifact_dir / "test_evaluation_receipt.json"
    # Exclusive creation precedes any test-file access; failures also consume the attempt.
    with receipt.open("x") as stream:
        json.dump(
            {
                "status": "started",
                "started_at": datetime.now(UTC).isoformat(),
                "configuration": configuration,
            },
            stream,
            indent=2,
        )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/transformer"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/transformer"))
    parser.add_argument("--comparison", type=Path, default=Path("reports/transformer_vs_baseline"))
    parser.add_argument("--timing-examples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--freeze",
        type=Path,
        help="After full validation, write frozen final-test configuration here",
    )
    parser.add_argument(
        "--freeze-file", type=Path, help="Required frozen configuration for official test"
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Consume the one official-test attempt for this checkpoint",
    )
    args = parser.parse_args()
    receipt = None
    try:
        if (
            args.batch_size < 1
            or args.timing_examples < args.batch_size
            or args.timing_examples % args.batch_size
        ):
            raise ValueError("Timing example count must be positive and divisible by batch size.")
        if args.warmup < 1 or args.repeats < 1:
            raise ValueError("Warmup and repeats must be positive.")
        if args.split == "validation" and (args.final_test or args.freeze_file):
            raise ValueError("Final-test flags apply only to --split test.")
        if args.split == "test" and (not args.final_test or not args.freeze_file or args.freeze):
            raise ValueError("Test requires --final-test --freeze-file and forbids --freeze.")
        seed_everything(threads=1)
        manifest_path = args.data_dir / "manifest.json"
        labels = canonical_labels(args.baseline_dir, manifest_path)
        metadata = json.loads((args.artifact_dir / "metadata.json").read_text())
        smoke = bool(metadata["smoke"])
        timing = {
            "examples": args.timing_examples,
            "batch_sizes": [1, args.batch_size],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cpu_threads": 1,
        }
        configuration = frozen_configuration(metadata, args.baseline_dir, manifest_path, timing)
        if (
            metadata["baseline_sha256"] != configuration["baseline_sha256"]
            or metadata["manifest_sha256"] != configuration["manifest_sha256"]
        ):
            raise ValueError("Baseline or dataset manifest differs from training provenance.")
        model = TransformerModel.load(args.artifact_dir, labels)
        baseline = Baseline.load(args.baseline_dir)
        if baseline.labels != labels:
            raise ValueError("Loaded baseline label order differs.")
        if args.split == "test":
            receipt = claim_final_test(args.artifact_dir, args.freeze_file, configuration, smoke)
            manifest = json.loads(manifest_path.read_text())
            test_path = args.data_dir / "test.parquet"
            if sha256_file(test_path) != manifest["splits"]["test"]["sha256"]:
                raise ValueError("Official test checksum mismatch; no rows may be filtered.")
            frame = read_data(test_path)
            if len(frame) != 3080 or set(frame["intent"]) != set(labels):
                raise ValueError("Expected all 3,080 official test examples and 77 labels.")
        else:
            _, frame, _, _ = development_data(args.data_dir, args.baseline_dir, smoke)
        if args.timing_examples > len(frame):
            raise ValueError("Too few evaluation examples for the requested timing sample.")
        texts = frame.sample(n=args.timing_examples, random_state=42)["text"].tolist()
        torch, _, _ = dependencies()
        torch.set_num_threads(1)
        report: dict[str, Any] = {
            "split": args.split,
            "rows": len(frame),
            "smoke": smoke,
            "model_version": metadata["model_version"],
            "configuration": configuration,
            "timing_configuration": timing,
            "created_at": datetime.now(UTC).isoformat(),
            "training_run": {
                key: metadata.get(key)
                for key in (
                    "configuration",
                    "random_seed",
                    "epochs_completed",
                    "selected_epoch",
                    "training_rows",
                    "validation_rows",
                    "resolved_model_revision",
                    "runtime",
                    "hardware",
                    "training_seconds",
                    "initialization_seconds",
                    "history",
                    "dataset_checksums",
                    "source_revision",
                    "label_revision",
                )
            },
        }
        for name, predict in (
            ("baseline", baseline.probabilities),
            ("transformer", partial(model.probabilities, batch_size=args.batch_size)),
        ):
            metrics = evaluate_probabilities(
                frame, predict(frame["text"].tolist()), labels, args.output_dir / args.split / name
            )
            size = (
                sum(
                    (args.baseline_dir / file).stat().st_size
                    for file in ("baseline.joblib", "metadata.json")
                )
                if name == "baseline"
                else sum(
                    (args.artifact_dir / file).stat().st_size
                    for file in [*metadata["checkpoint_files"], "metadata.json"]
                )
            )
            report[name] = {
                "metrics": metrics,
                "latency": cpu_timing(
                    predict, texts, (1, args.batch_size), args.warmup, args.repeats
                ),
                "artifact_bytes": size,
                "training_seconds": metadata["training_seconds"] if name == "transformer" else None,
                "training_hardware": metadata["hardware"] if name == "transformer" else None,
            }
        if (
            report["baseline"]["latency"]["examples_sha256"]
            != report["transformer"]["latency"]["examples_sha256"]
        ):
            raise ValueError("Timing examples differ between models.")
        write_comparison(report, args.comparison)
        if args.freeze:
            if smoke:
                raise ValueError("Smoke runs cannot freeze a final-test configuration.")
            args.freeze.parent.mkdir(parents=True, exist_ok=True)
            args.freeze.write_text(json.dumps(configuration, indent=2) + "\n")
        if receipt:
            receipt.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "completed_at": datetime.now(UTC).isoformat(),
                        "configuration": configuration,
                        "report": str(args.comparison.with_suffix(".json")),
                    },
                    indent=2,
                )
                + "\n"
            )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Transformer evaluation failed: {exc}\n")
    print(
        json.dumps(
            {
                "split": args.split,
                "rows": len(frame),
                "smoke": smoke,
                "comparison": str(args.comparison.with_suffix(".md")),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
