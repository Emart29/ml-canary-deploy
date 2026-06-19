import io
import uuid
from typing import Any

import joblib

from store.blob import BlobStore
from store.metadata import MetadataStore
from db.models import ModelVersion


class ModelRegistry:
    def __init__(self, metadata_store: MetadataStore, blob_store: BlobStore):
        self._meta = metadata_store
        self._blob = blob_store

    async def register(
        self,
        name: str,
        model_object: Any,
        framework: str = "sklearn",
        metrics: dict | None = None,
        parameters: dict | None = None,
        description: str = "",
        tags: list[str] | None = None,
        created_by: str = "system",
    ) -> ModelVersion:
        metrics = metrics or {}
        parameters = parameters or {}
        tags = tags or []

        version_number = await self._meta.get_next_version_number(name)
        storage_key = f"models/{name}/v{version_number}/{name}.joblib"

        buf = io.BytesIO()
        joblib.dump(model_object, buf)
        model_bytes = buf.getvalue()

        await self._blob.upload_bytes(storage_key, model_bytes, "application/octet-stream")

        return await self._meta.create_model_version(
            name=name,
            version=version_number,
            framework=framework,
            storage_path=storage_key,
            metrics=metrics,
            parameters=parameters,
            description=description,
            tags=tags,
            created_by=created_by,
        )

    async def load(self, name: str, version: int | None = None) -> tuple[Any, ModelVersion]:
        if version is None:
            mv = await self._meta.get_latest_model_version(name)
        else:
            mv = await self._meta.get_model_version_by_number(name, version)

        if mv is None:
            raise ValueError(f"Model '{name}' version {version or 'latest'} not found")

        model_bytes = await self._blob.download_bytes(mv.storage_path)
        model_object = joblib.load(io.BytesIO(model_bytes))
        return model_object, mv

    async def list_models(self) -> list[str]:
        return await self._meta.list_model_names()

    async def list_versions(self, name: str) -> list[ModelVersion]:
        return await self._meta.list_model_versions(name)

    async def get_version(self, name: str, version: int) -> ModelVersion | None:
        return await self._meta.get_model_version_by_number(name, version)

    async def delete_version(self, model_version_id: uuid.UUID) -> bool:
        mv = await self._meta.get_model_version(model_version_id)
        if mv is None:
            return False
        await self._blob.delete_object(mv.storage_path)
        await self._meta.delete_model_version(model_version_id)
        return True
