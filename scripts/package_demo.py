"""Create a local, allowlisted release directory; never upload or deploy it."""

import argparse
import json
import shutil
from pathlib import Path

from intentops.model import Baseline


def package_demo(root: Path, output: Path) -> list[str]:
    baseline = Baseline.load(root / "artifacts")
    if output.exists():
        raise ValueError(
            f"{output} already exists. Choose a new --output; existing files are preserved."
        )
    roots = [
        "README.md",
        "AGENTS.md",
        "LICENSE",
        "NOTICE.md",
        "pyproject.toml",
        "requirements-runtime.txt",
        "Makefile",
        "Dockerfile",
        ".dockerignore",
        ".env.example",
        "render.yaml",
        ".github/workflows/ci.yml",
        "artifacts/baseline.joblib",
        "artifacts/metadata.json",
        "reports/README.md",
        "reports/summary.json",
    ]
    paths = [root / name for name in roots]
    for directory, pattern in (
        ("src/intentops", "*.py"),
        ("src/intentops/templates", "*.html"),
        ("src/intentops/static", "*.css"),
        ("scripts", "*.py"),
        ("tests", "*.py"),
        ("docs", "*.md"),
        ("docs/images", "*.png"),
    ):
        paths.extend(sorted((root / directory).glob(pattern)))
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Release input must be an ordinary file: {path}")
    output.mkdir(parents=True)
    for path in paths:
        destination = output / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    # The release tracks its two frozen public-data artifacts without git add -f.
    (output / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.pyc\n.*_cache/\n.env\n.env.*\n!.env.example\n"
        "data/raw/*\n.coverage*\n*.egg-info/\ndemo-release/\n"
    )
    (output / "release.json").write_text(
        json.dumps(
            {
                "model_version": baseline.version,
                "model_sha256": baseline.metadata["model_sha256"],
                "files": sorted([str(path.relative_to(root)) for path in paths] + [".gitignore"]),
                "deployment_status": "prepared_locally_only",
            },
            indent=2,
        )
        + "\n"
    )
    return [str(path.relative_to(root)) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo-release"))
    args = parser.parse_args()
    try:
        paths = package_demo(Path.cwd(), args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")
    print(f"Prepared {len(paths)} reviewed inputs in {args.output}; nothing uploaded or deployed.")


if __name__ == "__main__":
    main()
