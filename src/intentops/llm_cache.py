"""Private experiment cache: hashed keys, interprocess locks and atomic replacement."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ResponseCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def resolve(
        self, key: str, create: Callable[[], dict[str, Any]]
    ) -> tuple[dict[str, Any], bool]:
        from filelock import FileLock

        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("Cache keys must be SHA-256 hex digests.")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        with FileLock(str(path) + ".lock", timeout=300):
            if path.exists():
                stored = json.loads(path.read_text())
                if stored["key"] != key or stored["sha256"] != fingerprint(stored["response"]):
                    raise ValueError("Cache checksum mismatch.")
                return dict(stored["response"]), True
            response = create()
            atomic_json(path, {"key": key, "sha256": fingerprint(response), "response": response})
            return response, False
