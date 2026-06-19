import asyncio
from datetime import datetime, timezone

from config import settings
from db.models import (
    HealthStatusEnum, RecommendationEnum, EventTypeEnum, ModelStatusEnum,
)
from store.metadata import MetadataStore
from core.deployment import DeploymentEngine
from core.metrics import CanaryMetrics
from db.base import AsyncSessionLocal

# Promote only once the canary has handled at least this many requests.
MIN_CANARY_REQUESTS_TO_PROMOTE = 100


class HealthChecker:
    def __init__(
        self, metadata_store: MetadataStore, deployment_engine: DeploymentEngine,
        metrics: CanaryMetrics,
    ):
        self._meta = metadata_store
        self._engine = deployment_engine
        self._metrics = metrics

    async def evaluate(self, deployment_name: str):
        deployment = await self._meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")

        now = datetime.now(timezone.utc)

        # No canary running -> trivially healthy, nothing to compare.
        if deployment.status != ModelStatusEnum.canary_running or not deployment.canary_model_id:
            return await self._meta.save_health_snapshot(
                deployment_id=deployment.id,
                checked_at=now,
                baseline_stats={},
                canary_stats={},
                health_status=HealthStatusEnum.healthy,
                recommendation=RecommendationEnum.continue_,
                details={"note": "no canary running"},
            )

        # Prometheus cumulative view.
        prom = self._metrics.get_metrics_snapshot(deployment_name)

        # Window-based view from PostgreSQL (last 5 min).
        baseline_err = await self._meta.get_error_rate(deployment.id, "baseline", 5)
        canary_err = await self._meta.get_error_rate(deployment.id, "canary", 5)
        baseline_lat = await self._meta.get_latency_percentiles(deployment.id, "baseline", 5)
        canary_lat = await self._meta.get_latency_percentiles(deployment.id, "canary", 5)
        baseline_count = await self._meta.get_request_count(deployment.id, "baseline", 5)
        canary_count = await self._meta.get_request_count(deployment.id, "canary", 5)

        baseline_stats = {
            "request_count": baseline_count,
            "error_rate": baseline_err,
            "p50": baseline_lat["p50"],
            "p95": baseline_lat["p95"],
        }
        canary_stats = {
            "request_count": canary_count,
            "error_rate": canary_err,
            "p50": canary_lat["p50"],
            "p95": canary_lat["p95"],
        }

        error_delta = canary_err - baseline_err
        latency_delta = canary_lat["p95"] - baseline_lat["p95"]

        crit_error = error_delta > settings.MAX_ERROR_RATE_DELTA
        crit_latency = latency_delta > settings.MAX_LATENCY_P95_DELTA_MS
        degraded_error = error_delta > (settings.MAX_ERROR_RATE_DELTA / 2)
        degraded_latency = latency_delta > (settings.MAX_LATENCY_P95_DELTA_MS / 2)

        if crit_error or crit_latency:
            health_status = HealthStatusEnum.critical
            recommendation = RecommendationEnum.rollback
        elif degraded_error or degraded_latency:
            health_status = HealthStatusEnum.degraded
            recommendation = RecommendationEnum.continue_
        else:
            health_status = HealthStatusEnum.healthy
            if canary_count >= MIN_CANARY_REQUESTS_TO_PROMOTE and canary_err < baseline_err:
                recommendation = RecommendationEnum.promote
            else:
                recommendation = RecommendationEnum.continue_

        details = {
            "error_delta": round(error_delta, 4),
            "latency_p95_delta_ms": round(latency_delta, 2),
            "thresholds": {
                "max_error_rate_delta": settings.MAX_ERROR_RATE_DELTA,
                "max_latency_p95_delta_ms": settings.MAX_LATENCY_P95_DELTA_MS,
            },
            "prometheus": prom,
            "reasons": {
                "critical_error_rate": crit_error,
                "critical_latency": crit_latency,
                "degraded_error_rate": degraded_error,
                "degraded_latency": degraded_latency,
            },
        }

        snapshot = await self._meta.save_health_snapshot(
            deployment_id=deployment.id,
            checked_at=now,
            baseline_stats=baseline_stats,
            canary_stats=canary_stats,
            health_status=health_status,
            recommendation=recommendation,
            details=details,
        )

        # Audit trail: log a pass/fail event alongside the snapshot.
        event_type = (
            EventTypeEnum.health_check_failed
            if health_status == HealthStatusEnum.critical
            else EventTypeEnum.health_check_passed
        )
        await self._meta.add_event(
            deployment_id=deployment.id,
            event_type=event_type,
            canary_traffic_pct=deployment.canary_traffic_pct,
            details={
                "health_status": health_status.value,
                "recommendation": recommendation.value,
                "error_delta": round(error_delta, 4),
                "latency_p95_delta_ms": round(latency_delta, 2),
            },
        )
        return snapshot

    async def run_auto_decision(self, deployment_name: str) -> str:
        snapshot = await self.evaluate(deployment_name)
        deployment = await self._meta.get_deployment_by_name(deployment_name)

        recommendation = snapshot.recommendation
        rec_value = recommendation.value if hasattr(recommendation, "value") else recommendation

        if rec_value == RecommendationEnum.rollback.value:
            if settings.AUTO_ROLLBACK_ENABLED and deployment.auto_rollback_enabled:
                await self._engine.rollback(deployment_name, reason="auto: health check critical")
                await self._meta.add_event(
                    deployment_id=deployment.id,
                    event_type=EventTypeEnum.auto_rollback_triggered,
                    canary_traffic_pct=0.0,
                    details={"trigger": "health_check", "health_snapshot_id": str(snapshot.id)},
                )
                return "rolled_back"
            return "no_action"

        if rec_value == RecommendationEnum.promote.value:
            await self._engine.promote(deployment_name, reason="auto: health check passed")
            return "promoted"

        return "no_action"


class HealthCheckScheduler:
    """Background loop that runs auto-decisions on all active canary deployments."""

    def __init__(self, health_checker: HealthChecker, interval_seconds: int | None = None):
        self._health_checker = health_checker
        self._interval = interval_seconds or settings.HEALTH_CHECK_INTERVAL_SECONDS
        self._task: asyncio.Task | None = None
        self._running = False

    async def _loop(self) -> None:
        # Reuse the stateless collaborators; only the DB session is rebuilt per
        # iteration so the background loop never shares a session with request handlers.
        metrics = self._health_checker._metrics
        registry = self._health_checker._engine._registry
        router = self._health_checker._engine._router
        while self._running:
            try:
                async with AsyncSessionLocal() as session:
                    meta = MetadataStore(session)
                    active = await meta.list_deployments(status=ModelStatusEnum.canary_running.value)
                    engine = DeploymentEngine(meta, registry, router)
                    checker = HealthChecker(meta, engine, metrics)
                    for deployment in active:
                        try:
                            await checker.run_auto_decision(deployment.name)
                        except Exception as exc:  # pragma: no cover - per-deployment guard
                            print(f"[health-scheduler] error checking {deployment.name}: {exc}")
            except Exception as exc:  # pragma: no cover - never let the loop die
                print(f"[health-scheduler] loop error: {exc}")
            await asyncio.sleep(self._interval)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
