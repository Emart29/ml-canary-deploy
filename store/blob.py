import asyncio
import io

import minio
from minio.error import S3Error

from config import settings


class BlobStore:
    def __init__(self):
        self._client = minio.Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        self._bucket = settings.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def _upload_bytes_sync(self, key: str, data: bytes, content_type: str) -> str:
        self._client.put_object(
            bucket_name=self._bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def _download_bytes_sync(self, key: str) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def _object_exists_sync(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error:
            return False

    async def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        return await asyncio.to_thread(self._upload_bytes_sync, key, data, content_type)

    async def download_bytes(self, key: str) -> bytes:
        return await asyncio.to_thread(self._download_bytes_sync, key)

    async def object_exists(self, key: str) -> bool:
        return await asyncio.to_thread(self._object_exists_sync, key)

    async def delete_object(self, key: str) -> None:
        await asyncio.to_thread(self._client.remove_object, self._bucket, key)

    async def list_objects(self, prefix: str = "") -> list[str]:
        def _list():
            return [obj.object_name for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True)]
        return await asyncio.to_thread(_list)
