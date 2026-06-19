import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
from sqlalchemy import select, update, func, and_, cast, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    ModelVersion, Deployment, DeploymentEvent, PredictionLog,
    HealthSnapshot, ModelStatusEnum, EventTypeEnum,
)


class MetadataStore:
    def __init__(self, session: AsyncSession):
        self._session = session

    # ------------------------------------------------------------------
    # Model versions
    # ------------------------------------------------------------------

    async def create_model_version(
        self, name: str, version: int, framework: str, storage_path: str,
        metrics: dict, parameters: dict, description: str, tags: list,
        created_by: str,
    ) -> ModelVersion:
        mv = ModelVersion(
            name=name, version=version, framework=framework,
            storage_path=storage_path, metrics=metrics,
            parameters=parameters, description=description,
            tags=tags, created_by=created_by,
        )
        self._session.add(mv)
        await self._session.commit()
        await self._session.refresh(mv)
        return mv

    async def get_model_version(self, version_id: uuid.UUID) -> ModelVersion | None:
        result = await self._session.execute(
            select(ModelVersion).where(ModelVersion.id == version_id)
        )
        return result.scalar_one_or_none()

    async def get_latest_model_version(self, name: str) -> ModelVersion | None:
        result = await self._session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == name)
            .order_by(ModelVersion.version.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_model_version_by_number(self, name: str, version: int) -> ModelVersion | None:
        result = await self._session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == name, ModelVersion.version == version)
        )
        return result.scalar_one_or_none()

    async def list_model_versions(self, name: str) -> list[ModelVersion]:
        result = await self._session.execute(
            select(ModelVersion)
            .where(ModelVersion.name == name)
            .order_by(ModelVersion.version.asc())
        )
        return list(result.scalars().all())

    async def list_model_names(self) -> list[str]:
        result = await self._session.execute(
            select(ModelVersion.name).distinct().order_by(ModelVersion.name)
        )
        return list(result.scalars().all())

    async def get_next_version_number(self, name: str) -> int:
        result = await self._session.execute(
            select(func.max(ModelVersion.version)).where(ModelVersion.name == name)
        )
        current_max = result.scalar_one_or_none()
        return (current_max or 0) + 1

    async def delete_model_version(self, version_id: uuid.UUID) -> None:
        mv = await self.get_model_version(version_id)
        if mv:
            await self._session.delete(mv)
            await self._session.commit()

    # ------------------------------------------------------------------
    # Deployments
    # ------------------------------------------------------------------

    async def create_deployment(
        self, name: str, baseline_model_id: uuid.UUID, created_by: str = "system",
    ) -> Deployment:
        d = Deployment(
            name=name,
            baseline_model_id=baseline_model_id,
            status=ModelStatusEnum.stable,
            created_by=created_by,
        )
        self._session.add(d)
        await self._session.commit()
        await self._session.refresh(d)
        return d

    async def get_deployment(self, deployment_id: uuid.UUID) -> Deployment | None:
        result = await self._session.execute(
            select(Deployment).where(Deployment.id == deployment_id)
        )
        return result.scalar_one_or_none()

    async def get_deployment_by_name(self, name: str) -> Deployment | None:
        result = await self._session.execute(
            select(Deployment).where(Deployment.name == name)
        )
        return result.scalar_one_or_none()

    async def list_deployments(self, status: str | None = None) -> list[Deployment]:
        q = select(Deployment).order_by(Deployment.started_at.desc())
        if status:
            q = q.where(Deployment.status == status)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def update_deployment(self, deployment_id: uuid.UUID, **kwargs) -> Deployment:
        kwargs["updated_at"] = datetime.now(timezone.utc)
        await self._session.execute(
            update(Deployment).where(Deployment.id == deployment_id).values(**kwargs)
        )
        await self._session.commit()
        return await self.get_deployment(deployment_id)

    async def start_canary(
        self, deployment_id: uuid.UUID, canary_model_id: uuid.UUID, initial_traffic_pct: float,
    ) -> Deployment:
        return await self.update_deployment(
            deployment_id,
            canary_model_id=canary_model_id,
            canary_traffic_pct=initial_traffic_pct,
            status=ModelStatusEnum.canary_running,
        )

    async def update_canary_traffic(self, deployment_id: uuid.UUID, traffic_pct: float) -> Deployment:
        return await self.update_deployment(deployment_id, canary_traffic_pct=traffic_pct)

    async def promote_canary(self, deployment_id: uuid.UUID) -> Deployment:
        d = await self.get_deployment(deployment_id)
        return await self.update_deployment(
            deployment_id,
            baseline_model_id=d.canary_model_id,
            canary_model_id=None,
            canary_traffic_pct=0.0,
            status=ModelStatusEnum.stable,
            promoted_at=datetime.now(timezone.utc),
        )

    async def rollback_canary(self, deployment_id: uuid.UUID, reason: str) -> Deployment:
        return await self.update_deployment(
            deployment_id,
            canary_model_id=None,
            canary_traffic_pct=0.0,
            status=ModelStatusEnum.stable,
            rolled_back_at=datetime.now(timezone.utc),
            rollback_reason=reason,
        )

    # ------------------------------------------------------------------
    # Deployment events
    # ------------------------------------------------------------------

    async def add_event(
        self, deployment_id: uuid.UUID, event_type: str,
        canary_traffic_pct: float, details: dict,
    ) -> DeploymentEvent:
        ev = DeploymentEvent(
            deployment_id=deployment_id,
            event_type=event_type,
            canary_traffic_pct=canary_traffic_pct,
            details=details,
        )
        self._session.add(ev)
        await self._session.commit()
        await self._session.refresh(ev)
        return ev

    async def get_events(self, deployment_id: uuid.UUID, limit: int = 50) -> list[DeploymentEvent]:
        result = await self._session.execute(
            select(DeploymentEvent)
            .where(DeploymentEvent.deployment_id == deployment_id)
            .order_by(DeploymentEvent.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Prediction logs
    # ------------------------------------------------------------------

    async def log_prediction(
        self, deployment_id: uuid.UUID, model_version_id: uuid.UUID,
        model_role: str, entity_id: str, input_hash: str,
        prediction: float, prediction_class: int | None, confidence: float | None,
        latency_ms: float, is_error: bool, error_message: str | None,
    ) -> PredictionLog:
        pl = PredictionLog(
            deployment_id=deployment_id,
            model_version_id=model_version_id,
            model_role=model_role,
            entity_id=entity_id,
            input_hash=input_hash,
            prediction=prediction,
            prediction_class=prediction_class,
            confidence=confidence,
            latency_ms=latency_ms,
            is_error=is_error,
            error_message=error_message,
        )
        self._session.add(pl)
        await self._session.commit()
        await self._session.refresh(pl)
        return pl

    async def get_predictions(
        self, deployment_id: uuid.UUID, model_role: str | None = None, limit: int = 100,
    ) -> list[PredictionLog]:
        q = (
            select(PredictionLog)
            .where(PredictionLog.deployment_id == deployment_id)
            .order_by(PredictionLog.predicted_at.desc())
            .limit(limit)
        )
        if model_role:
            q = q.where(PredictionLog.model_role == model_role)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def get_error_rate(
        self, deployment_id: uuid.UUID, model_role: str, window_minutes: int = 5,
    ) -> float:
        since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        result = await self._session.execute(
            select(
                func.count(PredictionLog.id).label("total"),
                func.sum(cast(PredictionLog.is_error, Integer)).label("errors"),
            ).where(
                and_(
                    PredictionLog.deployment_id == deployment_id,
                    PredictionLog.model_role == model_role,
                    PredictionLog.predicted_at >= since,
                )
            )
        )
        row = result.one()
        total = row.total or 0
        errors = row.errors or 0
        return (errors / total) if total > 0 else 0.0

    async def get_latency_percentiles(
        self, deployment_id: uuid.UUID, model_role: str, window_minutes: int = 5,
    ) -> dict:
        since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        result = await self._session.execute(
            select(PredictionLog.latency_ms).where(
                and_(
                    PredictionLog.deployment_id == deployment_id,
                    PredictionLog.model_role == model_role,
                    PredictionLog.predicted_at >= since,
                    PredictionLog.is_error == False,
                )
            )
        )
        latencies = [row[0] for row in result.all()]
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    async def get_request_count(
        self, deployment_id: uuid.UUID, model_role: str, window_minutes: int = 5,
    ) -> int:
        since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        result = await self._session.execute(
            select(func.count(PredictionLog.id)).where(
                and_(
                    PredictionLog.deployment_id == deployment_id,
                    PredictionLog.model_role == model_role,
                    PredictionLog.predicted_at >= since,
                )
            )
        )
        return result.scalar_one_or_none() or 0

    # ------------------------------------------------------------------
    # Health snapshots
    # ------------------------------------------------------------------

    async def save_health_snapshot(
        self, deployment_id: uuid.UUID, checked_at: datetime,
        baseline_stats: dict, canary_stats: dict,
        health_status: str, recommendation: str, details: dict,
    ) -> HealthSnapshot:
        hs = HealthSnapshot(
            deployment_id=deployment_id,
            checked_at=checked_at,
            baseline_request_count=baseline_stats.get("request_count", 0),
            baseline_error_rate=baseline_stats.get("error_rate", 0.0),
            baseline_latency_p50_ms=baseline_stats.get("p50", 0.0),
            baseline_latency_p95_ms=baseline_stats.get("p95", 0.0),
            canary_request_count=canary_stats.get("request_count", 0),
            canary_error_rate=canary_stats.get("error_rate", 0.0),
            canary_latency_p50_ms=canary_stats.get("p50", 0.0),
            canary_latency_p95_ms=canary_stats.get("p95", 0.0),
            health_status=health_status,
            recommendation=recommendation,
            details=details,
        )
        self._session.add(hs)
        await self._session.commit()
        await self._session.refresh(hs)
        return hs

    async def get_latest_health(self, deployment_id: uuid.UUID) -> HealthSnapshot | None:
        result = await self._session.execute(
            select(HealthSnapshot)
            .where(HealthSnapshot.deployment_id == deployment_id)
            .order_by(HealthSnapshot.checked_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_health_history(self, deployment_id: uuid.UUID, limit: int = 20) -> list[HealthSnapshot]:
        result = await self._session.execute(
            select(HealthSnapshot)
            .where(HealthSnapshot.deployment_id == deployment_id)
            .order_by(HealthSnapshot.checked_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
