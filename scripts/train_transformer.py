"""Fine-tune a transformer using official development training/validation only."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from intentops.transformer_model import (
    TransformerConfig,
    TransformerModel,
    development_data,
    seed_everything,
    train_transformer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="154 train / 77 validation rows, one CPU epoch, length 32, batch 8; not a benchmark",
    )
    args = parser.parse_args()
    try:
        output = args.artifact_dir or Path(
            "artifacts/transformer-smoke" if args.smoke else "artifacts/transformer"
        )
        if output.exists():
            raise ValueError(
                "Output already exists; choose a new artifact directory to preserve checkpoints."
            )
        config = TransformerConfig(
            model_name=args.model_name,
            revision=args.revision,
            epochs=1 if args.smoke else args.epochs,
            batch_size=8 if args.smoke else args.batch_size,
            max_length=32 if args.smoke else args.max_length,
            learning_rate=args.learning_rate,
            patience=args.patience,
            cpu_threads=args.cpu_threads,
            smoke=args.smoke,
        )
        hardware = seed_everything(config.seed, config.cpu_threads)
        device = (
            "cpu" if args.smoke else hardware["device"] if args.device == "auto" else args.device
        )
        if device == "cuda" and not hardware["cuda_available"]:
            raise ValueError(
                "CUDA requested but unavailable; use --smoke for a reduced CPU pipeline check."
            )
        hardware["device"] = device
        print(json.dumps({"hardware": hardware, "smoke": args.smoke}), flush=True)
        train, validation, labels, provenance = development_data(
            args.data_dir, args.baseline_dir, args.smoke
        )
        start = perf_counter()
        model = TransformerModel.build(config, labels, device, args.offline)
        initialization_seconds = perf_counter() - start
        metadata = train_transformer(model, train, validation, config)
        record = model.save(
            output,
            {
                **metadata,
                **provenance,
                "hardware": hardware,
                "initialization_seconds": initialization_seconds,
                "resolved_model_revision": getattr(model.model.config, "_commit_hash", None),
            },
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Transformer training failed: {exc}\n")
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "model_version": record["model_version"],
                "training_seconds": record["training_seconds"],
                "smoke": args.smoke,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
