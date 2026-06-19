from db.models import Deployment, ModelStatusEnum, EventTypeEnum
from store.metadata import MetadataStore
from core.registry import ModelRegistry
from core.router import TrafficRouter


class DeploymentEngine:
    def __init__(
        self, metadata_store: MetadataStore, registry: ModelRegistry, router: TrafficRouter,
    ):
        self._meta = metadata_store
        self._registry = registry
        self._router = router

    async def _sync_router(self, deployment: Deployment) -> None:
        """Push the current deployment state into Redis so the router sees it."""
        baseline = await self._meta.get_model_version(deployment.baseline_model_id)
        canary = None
        if deployment.canary_model_id:
            canary = await self._meta.get_model_version(deployment.canary_model_id)
        await self._router.set_deployment(deployment, baseline, canary)

    async def create_deployment(
        self, name: str, model_name: str, model_version: int | None = None,
        created_by: str = "system",
    ) -> Deployment:
        existing = await self._meta.get_deployment_by_name(name)
        if existing:
            raise ValueError(f"Deployment '{name}' already exists")

        if model_version is None:
            baseline = await self._meta.get_latest_model_version(model_name)
        else:
            baseline = await self._meta.get_model_version_by_number(model_name, model_version)
        if baseline is None:
            raise ValueError(f"Model '{model_name}' version {model_version or 'latest'} not found")

        deployment = await self._meta.create_deployment(
            name=name, baseline_model_id=baseline.id, created_by=created_by,
        )
        await self._sync_router(deployment)
        await self._meta.add_event(
            deployment_id=deployment.id,
            event_type=EventTypeEnum.canary_started,
            canary_traffic_pct=0.0,
            details={
                "action": "deployment_created",
                "baseline_model": baseline.name,
                "baseline_version": baseline.version,
            },
        )
        return deployment

    async def start_canary(
        self, deployment_name: str, canary_model_name: str,
        canary_version: int | None = None, initial_traffic_pct: float = 10.0,
        created_by: str = "system",
    ) -> Deployment:
        deployment = await self._meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")
        if deployment.status != ModelStatusEnum.stable:
            current = deployment.status.value if hasattr(deployment.status, "value") else deployment.status
            raise ValueError(
                f"Deployment '{deployment_name}' is '{current}', must be 'stable' to start a canary"
            )
        if not (0.0 < initial_traffic_pct <= 100.0):
            raise ValueError("initial_traffic_pct must be in (0, 100]")

        if canary_version is None:
            canary = await self._meta.get_latest_model_version(canary_model_name)
        else:
            canary = await self._meta.get_model_version_by_number(canary_model_name, canary_version)
        if canary is None:
            raise ValueError(f"Canary model '{canary_model_name}' version {canary_version or 'latest'} not found")

        deployment = await self._meta.start_canary(deployment.id, canary.id, initial_traffic_pct)
        await self._sync_router(deployment)
        await self._meta.add_event(
            deployment_id=deployment.id,
            event_type=EventTypeEnum.canary_started,
            canary_traffic_pct=initial_traffic_pct,
            details={
                "canary_model": canary.name,
                "canary_version": canary.version,
                "initial_traffic_pct": initial_traffic_pct,
                "canary_metrics": canary.metrics,
            },
        )
        return deployment

    async def adjust_traffic(self, deployment_name: str, new_canary_pct: float) -> Deployment:
        deployment = await self._meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")
        if deployment.status != ModelStatusEnum.canary_running:
            raise ValueError(f"Deployment '{deployment_name}' has no canary running")
        if not (0.0 <= new_canary_pct <= 100.0):
            raise ValueError("new_canary_pct must be between 0 and 100")

        deployment = await self._meta.update_canary_traffic(deployment.id, new_canary_pct)
        await self._sync_router(deployment)
        await self._meta.add_event(
            deployment_id=deployment.id,
            event_type=EventTypeEnum.traffic_adjusted,
            canary_traffic_pct=new_canary_pct,
            details={"new_canary_traffic_pct": new_canary_pct},
        )
        return deployment

    async def promote(self, deployment_name: str, reason: str = "manual") -> Deployment:
        deployment = await self._meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")
        if deployment.status != ModelStatusEnum.canary_running:
            raise ValueError(f"Deployment '{deployment_name}' has no canary to promote")

        promoted_canary = await self._meta.get_model_version(deployment.canary_model_id)
        deployment = await self._meta.promote_canary(deployment.id)
        await self._sync_router(deployment)
        await self._meta.add_event(
            deployment_id=deployment.id,
            event_type=EventTypeEnum.promoted,
            canary_traffic_pct=0.0,
            details={
                "reason": reason,
                "promoted_model": promoted_canary.name if promoted_canary else None,
                "promoted_version": promoted_canary.version if promoted_canary else None,
            },
        )
        return deployment

    async def rollback(self, deployment_name: str, reason: str = "manual") -> Deployment:
        deployment = await self._meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")
        if deployment.status != ModelStatusEnum.canary_running:
            raise ValueError(f"Deployment '{deployment_name}' has no canary to roll back")

        rolled_canary = await self._meta.get_model_version(deployment.canary_model_id)
        deployment = await self._meta.rollback_canary(deployment.id, reason)
        await self._sync_router(deployment)
        await self._meta.add_event(
            deployment_id=deployment.id,
            event_type=EventTypeEnum.rolled_back,
            canary_traffic_pct=0.0,
            details={
                "reason": reason,
                "rolled_back_model": rolled_canary.name if rolled_canary else None,
                "rolled_back_version": rolled_canary.version if rolled_canary else None,
            },
        )
        return deployment

    async def get_status(self, deployment_name: str) -> dict:
        deployment = await self._meta.get_deployment_by_name(deployment_name)
        if deployment is None:
            raise ValueError(f"Deployment '{deployment_name}' not found")

        baseline = await self._meta.get_model_version(deployment.baseline_model_id)
        canary = None
        if deployment.canary_model_id:
            canary = await self._meta.get_model_version(deployment.canary_model_id)

        latest_health = await self._meta.get_latest_health(deployment.id)
        recent_events = await self._meta.get_events(deployment.id, limit=5)

        return {
            "deployment": deployment,
            "baseline": baseline,
            "canary": canary,
            "traffic_split": {
                "baseline_pct": round(100.0 - deployment.canary_traffic_pct, 2),
                "canary_pct": round(deployment.canary_traffic_pct, 2),
            },
            "latest_health": latest_health,
            "recent_events": recent_events,
        }
