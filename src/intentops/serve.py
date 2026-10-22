"""Single-worker container/local launcher honoring the hosting platform's PORT."""

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535.")
    uvicorn.run(
        "intentops.api:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        access_log=False,
        limit_concurrency=16,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
