"""Train on a stratified subset of official training data; never open test data."""

import argparse
import json
from pathlib import Path

from intentops.config import DEFAULT_SEMANTIC_MODEL, TrainingConfig
from intentops.data import read_data
from intentops.model import train_baseline
from intentops.semantic import SemanticIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/train.parquet"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--c", type=float, default=4.0)
    parser.add_argument(
        "--semantic", action="store_true", help="Build embeddings; may download weights"
    )
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    args = parser.parse_args()
    try:
        baseline, train = train_baseline(
            read_data(args.data), args.artifact_dir, TrainingConfig(min_df=args.min_df, c=args.c)
        )
        if args.semantic:
            index = SemanticIndex.build(
                train["text"].tolist(),
                train["intent"].tolist(),
                args.semantic_model,
                baseline.version,
            )
            index.save(args.artifact_dir / "semantic.npz")
        else:
            # A previous index is invalidated by retraining; never accidentally compare stale data.
            (args.artifact_dir / "semantic.npz").unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    print(json.dumps(baseline.metadata, indent=2))


if __name__ == "__main__":
    main()
