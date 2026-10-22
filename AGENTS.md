# Repository guidance

- Use Python 3.11+ and the `src/intentops` layout.
- Add type hints to public interfaces; prefer small, straightforward functions.
- Keep dependencies and architecture minimal; no hosted APIs or infrastructure additions.
- Unit tests must use synthetic data and no network or downloaded embedding weights.
- Run `ruff check .`, `ruff format --check .`, `pytest`, and `mypy` after changes.
- Never log raw support text or fabricate quality/benchmark metrics.
- Preserve the official train/test separation. Fit on the development training subset only;
  choose configuration and thresholds on validation, never on official test labels.
- Keep optional semantic imports lazy and bind semantic artifacts to their baseline version.
