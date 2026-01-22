from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mlprg.contracts.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReloadResponse,
)
from mlprg.observability.logging import configure_logging, request_id_ctx_var
from mlprg.observability.metrics import model_reload_count, request_count, request_latency_seconds
from mlprg.observability.tracing import configure_tracing
from mlprg.registry.sqlite import SqliteRegistry
from mlprg.storage.local import LocalArtifactStore
from mlprg.storage.s3 import S3ArtifactStore
from mlprg.utils.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        self.registry = SqliteRegistry(settings.database_url)
        self.artifact_store = self._init_store()
        self.model: Any | None = None
        self.model_version: str | None = None
        self.metadata: dict[str, Any] = {}

    def _init_store(self) -> LocalArtifactStore | S3ArtifactStore:
        if settings.artifact_store == "s3":
            return S3ArtifactStore(
                bucket=settings.artifact_bucket,
                endpoint_url=settings.s3_endpoint_url,
                access_key=settings.s3_access_key,
                secret_key=settings.s3_secret_key,
            )
        return LocalArtifactStore(settings.artifact_local_path)

    def load_latest_production(self) -> None:
        record = self.registry.get_latest_by_status(settings.model_name, "PRODUCTION")
        if not record:
            logger.warning("No production model found")
            self.model = None
            self.model_version = None
            self.metadata = {}
            return
        artifacts = self.registry.list_artifacts(record.version)
        model_artifact = next((a for a in artifacts if a.uri.endswith("model.pkl")), None)
        if not model_artifact:
            raise RuntimeError("Model artifact missing")
        local_path = Path("/tmp") / f"{record.version}_model.pkl"
        self.artifact_store.get(model_artifact.uri, str(local_path))
        self.model = joblib.load(local_path)
        self.model_version = record.version
        self.metadata = record.metadata


model_service = ModelService()


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def create_app() -> FastAPI:
    configure_logging(settings.log_json)
    configure_tracing(settings.app_name)
    tracer = trace.get_tracer(__name__)
    app = FastAPI(title="MLPRG API", version="0.1.0")
    FastAPIInstrumentor.instrument_app(app)

    @app.middleware("http")
    async def metrics_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        token = request_id_ctx_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx_var.reset(token)
        latency = time.perf_counter() - start
        endpoint = request.url.path
        request_latency_seconds.labels(endpoint=endpoint).observe(latency)
        request_count.labels(endpoint=endpoint, status=str(response.status_code)).inc()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.on_event("startup")
    async def startup_event() -> None:
        model_service.load_latest_production()

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=HealthResponse)
    async def readyz() -> HealthResponse:
        if model_service.model is None:
            return HealthResponse(status="model_not_loaded")
        return HealthResponse(status="ready")

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionRequest) -> PredictionResponse:
        if model_service.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        with tracer.start_as_current_span("predict"):
            probabilities = model_service.model.predict_proba([payload.features])[0]
            prediction = int(probabilities[1] >= 0.5)
            return PredictionResponse(
                prediction=prediction,
                probability=float(probabilities[1]),
                model_version=model_service.model_version or "unknown",
            )

    @app.get("/model", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        if model_service.model_version is None:
            raise HTTPException(status_code=404, detail="No model loaded")
        return ModelInfoResponse(
            model_name=settings.model_name,
            model_version=model_service.model_version,
            status="PRODUCTION",
            metadata=model_service.metadata,
        )

    @app.post(
        "/admin/reload",
        response_model=ReloadResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def reload_model() -> ReloadResponse:
        with tracer.start_as_current_span("reload"):
            model_service.load_latest_production()
            model_reload_count.inc()
            if model_service.model_version is None:
                raise HTTPException(status_code=404, detail="No production model found")
            return ReloadResponse(reloaded=True, model_version=model_service.model_version)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
