import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    Integer, String, JSON, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelStatusEnum(str, enum.Enum):
    stable = "stable"
    canary_running = "canary_running"
    promoting = "promoting"
    rolling_back = "rolling_back"


class EventTypeEnum(str, enum.Enum):
    canary_started = "canary_started"
    traffic_adjusted = "traffic_adjusted"
    promoted = "promoted"
    rolled_back = "rolled_back"
    health_check_passed = "health_check_passed"
    health_check_failed = "health_check_failed"
    auto_rollback_triggered = "auto_rollback_triggered"


class ModelRoleEnum(str, enum.Enum):
    baseline = "baseline"
    canary = "canary"


class HealthStatusEnum(str, enum.Enum):
    healthy = "healthy"
    degraded = "degraded"
    critical = "critical"


class RecommendationEnum(str, enum.Enum):
    continue_ = "continue"
    promote = "promote"
    rollback = "rollback"


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    framework: Mapped[str] = mapped_column(String, nullable=False, default="sklearn")
    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="system")
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, nullable=False, default=list)


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    baseline_model_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    canary_model_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=True)
    canary_traffic_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(
        Enum(ModelStatusEnum, name="deployment_status_enum", create_type=True),
        nullable=False, default=ModelStatusEnum.stable
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rollback_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    auto_rollback_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="system")

    baseline_model: Mapped["ModelVersion"] = relationship("ModelVersion", foreign_keys=[baseline_model_id])
    canary_model: Mapped["ModelVersion | None"] = relationship("ModelVersion", foreign_keys=[canary_model_id])


class DeploymentEvent(Base):
    __tablename__ = "deployment_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("deployments.id"), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(
        Enum(EventTypeEnum, name="event_type_enum", create_type=True),
        nullable=False
    )
    canary_traffic_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PredictionLog(Base):
    __tablename__ = "prediction_logs_canary"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("deployments.id"), nullable=False, index=True)
    model_version_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False)
    model_role: Mapped[str] = mapped_column(
        Enum(ModelRoleEnum, name="model_role_enum", create_type=True),
        nullable=False
    )
    entity_id: Mapped[str] = mapped_column(String, nullable=False)
    input_hash: Mapped[str] = mapped_column(String, nullable=False)
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    prediction_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    is_error: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class HealthSnapshot(Base):
    __tablename__ = "health_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    deployment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("deployments.id"), nullable=False, index=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    baseline_request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    baseline_error_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    baseline_latency_p50_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    baseline_latency_p95_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    canary_request_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    canary_error_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    canary_latency_p50_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    canary_latency_p95_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    health_status: Mapped[str] = mapped_column(
        Enum(HealthStatusEnum, name="health_status_enum", create_type=True),
        nullable=False
    )
    recommendation: Mapped[str] = mapped_column(
        Enum(RecommendationEnum, name="recommendation_enum", create_type=True),
        nullable=False
    )
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
