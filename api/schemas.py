from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------

class CreateDeploymentRequest(BaseModel):
    name: str
    model_name: str
    model_version: int | None = None


class StartCanaryRequest(BaseModel):
    canary_model_name: str
    canary_model_version: int | None = None
    initial_traffic_pct: float = Field(default=10.0, gt=0.0, le=100.0)
    auto_rollback_enabled: bool = True


class AdjustTrafficRequest(BaseModel):
    canary_traffic_pct: float = Field(ge=0.0, le=100.0)


class PredictRequest(BaseModel):
    entity_id: str
    features: dict[str, float | int | str]


class RollbackRequest(BaseModel):
    reason: str = "manual"


class PromoteRequest(BaseModel):
    reason: str = "manual"


# ----------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------

class ModelVersionResponse(BaseModel):
    id: str
    name: str
    version: int
    framework: str
    metrics: dict[str, Any]
    created_at: datetime


class HealthSnapshotResponse(BaseModel):
    health_status: str
    recommendation: str
    baseline_stats: dict[str, Any]
    canary_stats: dict[str, Any]
    checked_at: datetime


class DeploymentResponse(BaseModel):
    id: str
    name: str
    status: str
    baseline: ModelVersionResponse | None = None
    canary: ModelVersionResponse | None = None
    canary_traffic_pct: float
    started_at: datetime
    latest_health: HealthSnapshotResponse | None = None


class PredictResponse(BaseModel):
    prediction_id: str
    model_role: str
    prediction: float
    prediction_class: int | None = None
    confidence: float | None = None
    latency_ms: float
    model_version: str


class DeploymentEventResponse(BaseModel):
    event_type: str
    canary_traffic_pct: float
    details: dict[str, Any]
    created_at: datetime
