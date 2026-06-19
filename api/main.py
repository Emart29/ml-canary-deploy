import hashlib
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Response
from fastapi.responses import PlainTextResponse

from config import settings
from db.base import AsyncSessionLocal, create_all_tables
from db.models import ModelVersion, Deployment, HealthSnapshot
from store.metadata import MetadataStore
from store.blob import BlobStore
from core.registry import ModelRegistry
from core.router import TrafficRouter, get_redis_client
from core.deployment import DeploymentEngine
from core.metrics import CanaryMetrics, generate_metrics_output, CONTENT_TYPE_LATEST
from core.health import HealthChecker, HealthCheckScheduler
from api.schemas import (
    CreateDeploymentRequest, StartCanaryRequest, AdjustTrafficRequest,
    PredictRequest, RollbackRequest, PromoteRequest,
    ModelVersionResponse, DeploymentResponse, PredictResponse,
    HealthSnapshotResponse, DeploymentEventResponse,
)


# ----------------------------------------------------------------------
# Shared singletons (created in lifespan)
# ----------------------------------------------------------------------

class AppState:
    blob: BlobStore
    redis: Any
    metrics: CanaryMetrics
    scheduler: HealthCheckScheduler
    # in-memory model cache: {model_version_id: fitted_model_object}
    model_cache: dict[str, Any]


state = AppState()
state.model_cache = {}


async def _load_model_cached(registry: ModelRegistry, model_version: ModelVersion):
    key = str(model_version.id)
    if key not in state.model_cache:
        model_obj, _ = await registry.load(model_version.name, model_version.version)
        state.model_cache[key] = model_obj
    return state.model_cache[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_all_tables()
    state.blob = BlobStore()
    state.redis = get_redis_client()
    state.metrics = CanaryMetrics()

    # Health-check scheduler runs against fresh sessions internally.
    async with AsyncSessionLocal() as session:
        meta = MetadataStore(session)
        registry = ModelRegistry(meta, state.blob)
        router = TrafficRouter(state.redis)
        engine = DeploymentEngine(meta, registry, router)
        checker = HealthChecker(meta, engine, state.metrics)
    state.scheduler = HealthCheckScheduler(checker, settings.HEALTH_CHECK_INTERVAL_SECONDS)
    await state.scheduler.start()
    try:
        yield
    finally:
        await state.scheduler.stop()


app = FastAPI(title="ML Canary Deploy", version="0.1.0", lifespan=lifespan)


# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session


def get_blob() -> BlobStore:
    return state.blob


def get_router() -> TrafficRouter:
    return TrafficRouter(state.redis)


def get_metrics() -> CanaryMetrics:
    return state.metrics


# ----------------------------------------------------------------------
# Serializers
# ----------------------------------------------------------------------

def _model_to_response(mv: ModelVersion | None) -> ModelVersionResponse | None:
    if mv is None:
        return None
    return ModelVersionResponse(
        id=str(mv.id), name=mv.name, version=mv.version,
        framework=mv.framework, metrics=mv.metrics, created_at=mv.created_at,
    )


def _health_to_response(hs: HealthSnapshot | None) -> HealthSnapshotResponse | None:
    if hs is None:
        return None
    status = hs.health_status.value if hasattr(hs.health_status, "value") else hs.health_status
    rec = hs.recommendation.value if hasattr(hs.recommendation, "value") else hs.recommendation
    return HealthSnapshotResponse(
        health_status=status,
        recommendation=rec,
        baseline_stats={
            "request_count": hs.baseline_request_count,
            "error_rate": hs.baseline_error_rate,
            "latency_p50_ms": hs.baseline_latency_p50_ms,
            "latency_p95_ms": hs.baseline_latency_p95_ms,
        },
        canary_stats={
            "request_count": hs.canary_request_count,
            "error_rate": hs.canary_error_rate,
            "latency_p50_ms": hs.canary_latency_p50_ms,
            "latency_p95_ms": hs.canary_latency_p95_ms,
        },
        checked_at=hs.checked_at,
    )


async def _deployment_to_response(meta: MetadataStore, d: Deployment) -> DeploymentResponse:
    baseline = await meta.get_model_version(d.baseline_model_id)
    canary = await meta.get_model_version(d.canary_model_id) if d.canary_model_id else None
    latest_health = await meta.get_latest_health(d.id)
    status = d.status.value if hasattr(d.status, "value") else d.status
    return DeploymentResponse(
        id=str(d.id), name=d.name, status=status,
        baseline=_model_to_response(baseline),
        canary=_model_to_response(canary),
        canary_traffic_pct=d.canary_traffic_pct,
        started_at=d.started_at,
        latest_health=_health_to_response(latest_health),
    )


def _hash_features(features: dict) -> str:
    payload = json.dumps(features, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


# ----------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics_endpoint():
    return Response(content=generate_metrics_output(), media_type=CONTENT_TYPE_LATEST)


# ----------------------------------------------------------------------
# Model registry
# ----------------------------------------------------------------------

@app.get("/models")
async def list_models(session=Depends(get_session)):
    meta = MetadataStore(session)
    return {"models": await meta.list_model_names()}


@app.get("/models/{name}/versions")
async def list_model_versions(name: str, session=Depends(get_session)):
    meta = MetadataStore(session)
    versions = await meta.list_model_versions(name)
    return {"name": name, "versions": [_model_to_response(v) for v in versions]}


@app.get("/models/{name}/versions/{version}")
async def get_model_version(name: str, version: int, session=Depends(get_session)):
    meta = MetadataStore(session)
    mv = await meta.get_model_version_by_number(name, version)
    if mv is None:
        raise HTTPException(404, f"Model '{name}' v{version} not found")
    resp = _model_to_response(mv)
    return {**resp.model_dump(), "parameters": mv.parameters, "description": mv.description, "tags": mv.tags}


# ----------------------------------------------------------------------
# Deployments
# ----------------------------------------------------------------------

@app.get("/deployments")
async def list_deployments(status: str | None = None, session=Depends(get_session)):
    meta = MetadataStore(session)
    deployments = await meta.list_deployments(status=status)
    return {"deployments": [await _deployment_to_response(meta, d) for d in deployments]}


@app.post("/deployments")
async def create_deployment(req: CreateDeploymentRequest, session=Depends(get_session)):
    meta = MetadataStore(session)
    engine = DeploymentEngine(meta, ModelRegistry(meta, state.blob), get_router())
    try:
        d = await engine.create_deployment(req.name, req.model_name, req.model_version)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # set traffic gauge to 0 on creation
    state.metrics.update_traffic_split(d.name, 0.0)
    baseline = await meta.get_model_version(d.baseline_model_id)
    if baseline and "accuracy" in (baseline.metrics or {}):
        state.metrics.set_model_accuracy(d.name, f"v{baseline.version}", "baseline", baseline.metrics["accuracy"])
    return await _deployment_to_response(meta, d)


@app.get("/deployments/{name}")
async def get_deployment(name: str, session=Depends(get_session)):
    meta = MetadataStore(session)
    d = await meta.get_deployment_by_name(name)
    if d is None:
        raise HTTPException(404, f"Deployment '{name}' not found")
    return await _deployment_to_response(meta, d)


@app.post("/deployments/{name}/canary")
async def start_canary(name: str, req: StartCanaryRequest, session=Depends(get_session)):
    meta = MetadataStore(session)
    engine = DeploymentEngine(meta, ModelRegistry(meta, state.blob), get_router())
    try:
        d = await engine.start_canary(
            name, req.canary_model_name, req.canary_model_version, req.initial_traffic_pct,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not req.auto_rollback_enabled:
        d = await meta.update_deployment(d.id, auto_rollback_enabled=False)
    state.metrics.update_traffic_split(d.name, d.canary_traffic_pct)
    canary = await meta.get_model_version(d.canary_model_id)
    if canary and "accuracy" in (canary.metrics or {}):
        state.metrics.set_model_accuracy(d.name, f"v{canary.version}", "canary", canary.metrics["accuracy"])
    return await _deployment_to_response(meta, d)


@app.patch("/deployments/{name}/traffic")
async def adjust_traffic(name: str, req: AdjustTrafficRequest, session=Depends(get_session)):
    meta = MetadataStore(session)
    engine = DeploymentEngine(meta, ModelRegistry(meta, state.blob), get_router())
    try:
        d = await engine.adjust_traffic(name, req.canary_traffic_pct)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state.metrics.update_traffic_split(d.name, d.canary_traffic_pct)
    return await _deployment_to_response(meta, d)


@app.post("/deployments/{name}/promote")
async def promote(name: str, req: PromoteRequest, session=Depends(get_session)):
    meta = MetadataStore(session)
    engine = DeploymentEngine(meta, ModelRegistry(meta, state.blob), get_router())
    try:
        d = await engine.promote(name, req.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state.metrics.update_traffic_split(d.name, 0.0)
    return await _deployment_to_response(meta, d)


@app.post("/deployments/{name}/rollback")
async def rollback(name: str, req: RollbackRequest, session=Depends(get_session)):
    meta = MetadataStore(session)
    engine = DeploymentEngine(meta, ModelRegistry(meta, state.blob), get_router())
    try:
        d = await engine.rollback(name, req.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state.metrics.update_traffic_split(d.name, 0.0)
    return await _deployment_to_response(meta, d)


@app.get("/deployments/{name}/events")
async def get_events(name: str, limit: int = 50, session=Depends(get_session)):
    meta = MetadataStore(session)
    d = await meta.get_deployment_by_name(name)
    if d is None:
        raise HTTPException(404, f"Deployment '{name}' not found")
    events = await meta.get_events(d.id, limit=limit)
    return {
        "deployment": name,
        "events": [
            DeploymentEventResponse(
                event_type=e.event_type.value if hasattr(e.event_type, "value") else e.event_type,
                canary_traffic_pct=e.canary_traffic_pct,
                details=e.details,
                created_at=e.created_at,
            )
            for e in events
        ],
    }


@app.get("/deployments/{name}/health")
async def get_health(name: str, limit: int = 20, session=Depends(get_session)):
    meta = MetadataStore(session)
    d = await meta.get_deployment_by_name(name)
    if d is None:
        raise HTTPException(404, f"Deployment '{name}' not found")
    latest = await meta.get_latest_health(d.id)
    history = await meta.get_health_history(d.id, limit=limit)
    return {
        "deployment": name,
        "latest": _health_to_response(latest),
        "history": [_health_to_response(h) for h in history],
    }


# ----------------------------------------------------------------------
# Prediction hot path
# ----------------------------------------------------------------------

@app.post("/predict/{deployment_name}", response_model=PredictResponse)
async def predict(deployment_name: str, req: PredictRequest, session=Depends(get_session)):
    meta = MetadataStore(session)
    registry = ModelRegistry(meta, state.blob)
    router = get_router()

    config = await router.get_deployment_config_by_name(deployment_name, meta)
    if config is None:
        raise HTTPException(404, f"Deployment '{deployment_name}' not found")

    role = router.route(config)

    if role == "canary" and config.get("canary_model_id"):
        model_name = config["canary_model_name"]
        model_version_num = int(config["canary_model_version"])
    else:
        role = "baseline"
        model_name = config["baseline_model_name"]
        model_version_num = int(config["baseline_model_version"])

    mv = await meta.get_model_version_by_number(model_name, model_version_num)
    if mv is None:
        raise HTTPException(500, f"Model {model_name} v{model_version_num} missing from registry")

    deployment_id = config["deployment_id"]
    input_hash = _hash_features(req.features)
    version_str = f"v{mv.version}"

    start = time.perf_counter()
    is_error = False
    error_message = None
    prediction_val = 0.0
    prediction_class = None
    confidence = None

    try:
        model = await _load_model_cached(registry, mv)
        X = pd.DataFrame([req.features])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            prediction_class = int(np.argmax(proba))
            confidence = float(np.max(proba))
            prediction_val = float(proba[-1])
        else:
            pred = model.predict(X)[0]
            prediction_val = float(pred)
            prediction_class = int(pred)
    except Exception as exc:
        is_error = True
        error_message = str(exc)[:480]

    latency_ms = (time.perf_counter() - start) * 1000.0

    state.metrics.record_prediction(deployment_name, version_str, role, latency_ms, is_error)

    log = await meta.log_prediction(
        deployment_id=deployment_id,
        model_version_id=mv.id,
        model_role=role,
        entity_id=req.entity_id,
        input_hash=input_hash,
        prediction=prediction_val,
        prediction_class=prediction_class,
        confidence=confidence,
        latency_ms=latency_ms,
        is_error=is_error,
        error_message=error_message,
    )

    if is_error:
        raise HTTPException(
            status_code=422,
            detail={
                "prediction_id": str(log.id), "model_role": role,
                "error": error_message, "latency_ms": round(latency_ms, 2),
                "model_version": version_str,
            },
        )

    return PredictResponse(
        prediction_id=str(log.id),
        model_role=role,
        prediction=prediction_val,
        prediction_class=prediction_class,
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
        model_version=version_str,
    )
