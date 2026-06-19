import random
from datetime import datetime, timezone

import redis

from config import settings
from db.models import Deployment, ModelVersion

KEY_PREFIX = "canary_deploy:"


def get_redis_client() -> redis.Redis:
    """Sync Redis client from settings — used in the per-request hot path."""
    return redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


class TrafficRouter:
    """Reads/writes deployment routing config in Redis so traffic-split changes
    take effect instantly without restarting the serving layer."""

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    @staticmethod
    def _key(deployment_id: str) -> str:
        return f"{KEY_PREFIX}{deployment_id}"

    async def set_deployment(
        self, deployment: Deployment, baseline: ModelVersion, canary: ModelVersion | None,
    ) -> None:
        """Write the full deployment config to Redis. Call on every change."""
        config = {
            "deployment_id": str(deployment.id),
            "deployment_name": deployment.name,
            "baseline_model_id": str(baseline.id),
            "baseline_model_name": baseline.name,
            "baseline_model_version": str(baseline.version),
            "canary_model_id": str(canary.id) if canary else "",
            "canary_model_name": canary.name if canary else "",
            "canary_model_version": str(canary.version) if canary else "0",
            "canary_traffic_pct": str(deployment.canary_traffic_pct),
            "status": str(deployment.status.value if hasattr(deployment.status, "value") else deployment.status),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._redis.hset(self._key(str(deployment.id)), mapping=config)

    async def get_deployment_config(self, deployment_id: str) -> dict | None:
        data = self._redis.hgetall(self._key(deployment_id))
        return data if data else None

    async def get_deployment_config_by_name(self, deployment_name: str, meta) -> dict | None:
        """Resolve a deployment's routing config by name. Reads Redis first;
        if the config is missing (e.g. Redis was flushed), rebuilds it from the
        database and re-caches it so the hot path stays resilient."""
        deployment = await meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            return None
        config = await self.get_deployment_config(str(deployment.id))
        if config is None:
            baseline = await meta.get_model_version(deployment.baseline_model_id)
            canary = None
            if deployment.canary_model_id:
                canary = await meta.get_model_version(deployment.canary_model_id)
            await self.set_deployment(deployment, baseline, canary)
            config = await self.get_deployment_config(str(deployment.id))
        return config

    def route(self, deployment_config: dict) -> str:
        """Pure, no-I/O routing decision. Returns 'baseline' or 'canary'.
        Called inline on every prediction request — keep it cheap."""
        if not deployment_config:
            return "baseline"
        canary_model_id = deployment_config.get("canary_model_id", "")
        if not canary_model_id:
            return "baseline"
        try:
            canary_pct = float(deployment_config.get("canary_traffic_pct", 0.0))
        except (TypeError, ValueError):
            canary_pct = 0.0
        if canary_pct <= 0.0:
            return "baseline"
        if canary_pct >= 100.0:
            return "canary"
        return "canary" if random.random() < (canary_pct / 100.0) else "baseline"

    async def list_active_deployments(self) -> list[dict]:
        configs = []
        for key in self._redis.scan_iter(match=f"{KEY_PREFIX}*"):
            data = self._redis.hgetall(key)
            if data:
                configs.append(data)
        return configs

    async def remove_deployment(self, deployment_id: str) -> None:
        self._redis.delete(self._key(deployment_id))
