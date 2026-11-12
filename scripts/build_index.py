"""Build only from verified fitting rows; never opens the official test split."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from intentops.llm_cache import atomic_json
from intentops.llm_path import LocalEncoder, chroma_client, index_binding, load_config
from intentops.transformer_model import development_data


def main() -> None:
    from filelock import FileLock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("../data/raw"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/llm-pilot.json"))
    parser.add_argument(
        "--descriptions", type=Path, default=Path("configs/intent-descriptions-v1.json")
    )
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/llm/index"))
    parser.add_argument("--embedding-cache", type=Path, default=Path(".hf-cache"))
    args = parser.parse_args()
    config = load_config(args.config)
    train, validation, labels, provenance = development_data(args.data_dir, args.baseline_dir)
    if len(train) != 8002 or len(validation) != 2001 or set(train.index) & set(validation.index):
        raise ValueError("Unexpected or overlapping development split.")
    descriptions = json.loads(args.descriptions.read_text())
    if set(descriptions) != set(labels) or any(
        not value.strip() for value in descriptions.values()
    ):
        raise ValueError("Descriptions must cover exactly the 77 canonical intent names.")
    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.index_dir) + ".lock", timeout=1):
        if args.index_dir.exists():
            raise ValueError("Index path exists; preserve it and use a new path for another build.")
        start = perf_counter()
        encoder = LocalEncoder(config, args.embedding_cache)
        client = chroma_client(args.index_dir)
        examples = client.create_collection(
            "fitting_examples", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        for offset in range(0, len(train), 256):
            batch = train.iloc[offset : offset + 256]
            texts = batch["text"].tolist()
            examples.add(
                ids=[str(i) for i in batch.index],
                documents=texts,
                metadatas=[{"intent": intent} for intent in batch["intent"]],
                embeddings=encoder.encode(texts).tolist(),
            )
        collection = client.create_collection(
            "intent_descriptions", embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        texts = [descriptions[label] for label in labels]
        collection.add(ids=labels, documents=texts, embeddings=encoder.encode(texts).tolist())
        binding = index_binding(config, labels, provenance, args.descriptions, train.index.tolist())
        atomic_json(args.index_dir / "binding.json", binding)
        atomic_json(
            args.index_dir / "build.json",
            {
                "configuration": asdict(config),
                "seconds": perf_counter() - start,
                "binding": binding,
            },
        )
    print(
        json.dumps(
            {
                "fitting_rows": examples.count(),
                "description_rows": collection.count(),
                "index": str(args.index_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
