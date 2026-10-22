"""FastAPI application factory with a clear missing-artifact health state."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from intentops.config import Settings
from intentops.demo import EXAMPLES, render_demo
from intentops.http import BodyLimitMiddleware
from intentops.logging import configure_logging
from intentops.model import Baseline
from intentops.router import Router
from intentops.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    MetadataResponse,
    Prediction,
    PredictRequest,
)
from intentops.semantic import SemanticIndex


def create_app(settings: Settings | None = None, router: Router | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.router = router
        app.state.load_error = None
        if router is None:
            try:
                baseline = Baseline.load(settings.artifact_dir)
                semantic = None
                if settings.semantic_enabled:
                    semantic = SemanticIndex.load(
                        settings.artifact_dir / "semantic.npz", baseline.version
                    )
                    # Local cache only. Explicit training is the only download path.
                    semantic.ensure_encoder()
                app.state.router = Router(baseline, settings, semantic)
            except Exception as exc:
                # Untrusted exceptions can include paths/data; expose actionable fixed text.
                # Startup is a boundary: retain a queryable failed-readiness state and
                # record the exception type without accidentally logging dataset text.
                logging.getLogger("intentops.startup").error(
                    "artifact_load_failed", extra={"event": {"exception_type": type(exc).__name__}}
                )
                app.state.load_error = (
                    "Could not load model artifacts. Run python scripts/train.py; check artifact "
                    "checksums and package versions. For semantic mode, install .[semantic], "
                    "run train.py --semantic, and retain the local encoder cache."
                )
        yield

    app = FastAPI(title="IntentOps", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.mount("/static", StaticFiles(directory=str(files("intentops").joinpath("static"))))

    def demo_response(body: str, status_code: int = 200) -> HTMLResponse:
        return HTMLResponse(
            body,
            status_code=status_code,
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'self'; form-action 'self'; "
                    "base-uri 'none'; frame-ancestors 'none'"
                ),
            },
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo(request: Request, example: str = "") -> HTMLResponse:
        service = request.app.state.router
        error = (
            "Model unavailable. Check /health and load trained artifacts."
            if service is None
            else None
        )
        return demo_response(render_demo(service, EXAMPLES.get(example, ""), error=error))

    @app.post("/demo", response_class=HTMLResponse, include_in_schema=False)
    async def demo_predict(request: Request) -> HTMLResponse:
        service = request.app.state.router
        if service is None:
            return demo_response(render_demo(None, error="Model unavailable. Check /health."), 503)
        if (
            request.headers.get("content-type", "").split(";")[0]
            != "application/x-www-form-urlencoded"
        ):
            return demo_response(
                render_demo(service, error="Submit the support-message form."), 415
            )
        try:
            fields = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
            if set(fields) != {"text"} or len(fields["text"]) != 1:
                raise ValueError("Expected one text field.")
            body = PredictRequest(text=fields["text"][0])
        except (ValidationError, ValueError, UnicodeDecodeError):
            return demo_response(
                render_demo(service, error="Enter 1–2,000 nonblank characters."), 422
            )
        prediction = await run_in_threadpool(service.predict, body.text)
        return demo_response(render_demo(service, body.text, prediction))

    def get_router(request: Request) -> Router:
        loaded: Router | None = request.app.state.router
        if loaded is None:
            raise HTTPException(status_code=503, detail=request.app.state.load_error)
        return loaded

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> JSONResponse:
        loaded = request.app.state.router is not None
        body = HealthResponse(
            status="ok" if loaded else "unavailable",
            model_loaded=loaded,
            detail=request.app.state.load_error,
        )
        return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(request: Request) -> MetadataResponse:
        service = get_router(request)
        config = service.settings
        return MetadataResponse(
            model_name=service.baseline.metadata["model_name"],
            model_version=service.baseline.version,
            class_count=len(service.baseline.labels),
            review_threshold=config.review_threshold,
            semantic_available=service.semantic is not None,
            semantic_enabled=config.semantic_enabled,
            semantic_min_similarity=config.semantic_min_similarity,
            semantic_min_vote=config.semantic_min_vote,
        )

    @app.post("/predict", response_model=Prediction, response_model_exclude_none=True)
    def predict(body: PredictRequest, request: Request) -> Prediction:
        return get_router(request).predict(body.text)

    @app.post("/batch-predict", response_model=BatchResponse, response_model_exclude_none=True)
    def batch_predict(body: BatchRequest, request: Request) -> BatchResponse:
        service = get_router(request)
        return BatchResponse(predictions=[service.predict(text) for text in body.texts])

    return app


app = create_app()
