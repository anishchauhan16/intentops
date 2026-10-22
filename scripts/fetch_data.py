"""Fetch the published PolyAI BANKING77 files through Hugging Face Datasets."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from intentops.data import sha256_file, validate_frame

DATASET_NAME = "PolyAI/banking77"
UPSTREAM_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
LABEL_REVISION = "90d4e2e"
LABEL_URL = (
    f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/{LABEL_REVISION}/dataset_infos.json"
)
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{UPSTREAM_REVISION}/banking_data"
)


def fetch_data(output_dir: Path) -> dict[str, Any]:
    try:
        from datasets import ClassLabel, Features, Value, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'Install data download support: python -m pip install -e ".[data]"'
        ) from exc
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(LABEL_URL, timeout=60) as response:
            names = json.load(response)["default"]["features"]["label"]["names"]
        if len(names) != 77 or len(set(names)) != 77:
            raise ValueError("Expected 77 unique BANKING77 intent names.")
        # Current datasets releases cannot execute the original banking77.py loader.
        # Use the same original files with the packaged CSV builder; no remote code.
        dataset = load_dataset(
            "csv",
            data_files={split: f"{SOURCE_ROOT}/{split}.csv" for split in ("train", "test")},
            features=Features({"text": Value("string"), "category": ClassLabel(names=names)}),
        )
        manifest: dict[str, Any] = {
            "dataset_name": DATASET_NAME,
            "source_revision": UPSTREAM_REVISION,
            "label_revision": LABEL_REVISION,
            "source_files": {split: f"{SOURCE_ROOT}/{split}.csv" for split in ("train", "test")},
            "intent_names": names,
            "splits": {},
        }
        # Finish both files before publishing a new bundle. Preserve source row order,
        # text, labels, and duplicate rows; never clean the official test split.
        with tempfile.TemporaryDirectory(dir=output_dir) as staging:
            for split, expected_rows in (("train", 10003), ("test", 3080)):
                frame = dataset[split].to_pandas().rename(columns={"category": "label"})
                frame["intent"] = frame["label"].map(dict(enumerate(names)))
                frame = frame[["text", "label", "intent"]]
                validate_frame(frame)
                if len(frame) != expected_rows:
                    raise ValueError(
                        f"Expected {expected_rows} {split} rows, received {len(frame)}."
                    )
                path = Path(staging) / f"{split}.parquet"
                frame.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
                manifest["splits"][split] = {
                    "rows": len(frame),
                    "file": path.name,
                    "sha256": sha256_file(path),
                }
            for split in ("train", "test"):
                shutil.move(
                    str(Path(staging) / f"{split}.parquet"), output_dir / f"{split}.parquet"
                )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except Exception as exc:
        raise RuntimeError(
            f"BANKING77 download/serialization failed ({type(exc).__name__}): {exc}. "
            "Check network access to huggingface.co and raw.githubusercontent.com, "
            "install .[data], "
            "and verify the output directory is writable. No rows are intentionally filtered."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    try:
        manifest = fetch_data(args.output_dir)
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
