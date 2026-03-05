"""
Storage Abstraction Layer

Provides a unified interface for storing raw match JSONs and model artifacts.
Backend is selected by STORAGE_BACKEND env var:
  - "local" (default): stores files on local disk
  - "s3": stores files in AWS S3

Switching from local to S3 requires only changing STORAGE_BACKEND and
adding AWS credentials in .env. No code changes anywhere else.
"""

import os
import orjson
import logging
from abc import ABC, abstractmethod

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract interface — all storage backends implement these methods."""

    @abstractmethod
    def store_raw_match(self, match_id: int, data: dict, source: str = "opendota") -> str:
        """Store raw match JSON. Returns the storage key/path."""
        ...

    @abstractmethod
    def get_raw_match(self, match_id: int, source: str = "opendota") -> dict | None:
        """Retrieve raw match JSON. Returns None if not found."""
        ...

    @abstractmethod
    def store_model(self, model_name: str, version: str, model_bytes: bytes) -> str:
        """Store a trained model artifact. Returns the storage key/path."""
        ...

    @abstractmethod
    def get_model(self, model_name: str, version: str) -> bytes | None:
        """Retrieve a model artifact. Returns None if not found."""
        ...

    @abstractmethod
    def list_model_versions(self, model_name: str) -> list[str]:
        """List all versions of a model."""
        ...


# ── Local Filesystem Backend ──────────────────────────────

class LocalStorageBackend(StorageBackend):
    """Stores files on local disk. Good for development, no cloud dependency."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _ensure_dir(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def store_raw_match(self, match_id: int, data: dict, source: str = "opendota") -> str:
        key = f"raw/{source}/{match_id}.json"
        full_path = os.path.join(self.base_dir, key)
        self._ensure_dir(full_path)
        with open(full_path, "wb") as f:
            f.write(orjson.dumps(data))
        logger.debug(f"Stored raw match {match_id} at {full_path}")
        return key

    def get_raw_match(self, match_id: int, source: str = "opendota") -> dict | None:
        key = f"raw/{source}/{match_id}.json"
        full_path = os.path.join(self.base_dir, key)
        if not os.path.exists(full_path):
            return None
        with open(full_path, "rb") as f:
            return orjson.loads(f.read())

    def store_model(self, model_name: str, version: str, model_bytes: bytes) -> str:
        key = f"models/{model_name}/{version}.pkl"
        full_path = os.path.join(self.base_dir, key)
        self._ensure_dir(full_path)
        with open(full_path, "wb") as f:
            f.write(model_bytes)
        logger.debug(f"Stored model {model_name}/{version} at {full_path}")
        return key

    def get_model(self, model_name: str, version: str) -> bytes | None:
        key = f"models/{model_name}/{version}.pkl"
        full_path = os.path.join(self.base_dir, key)
        if not os.path.exists(full_path):
            return None
        with open(full_path, "rb") as f:
            return f.read()

    def list_model_versions(self, model_name: str) -> list[str]:
        model_dir = os.path.join(self.base_dir, "models", model_name)
        if not os.path.exists(model_dir):
            return []
        return [
            f.replace(".pkl", "")
            for f in os.listdir(model_dir)
            if f.endswith(".pkl")
        ]


# ── S3 Backend ─────────────────────────────────────────────

class S3StorageBackend(StorageBackend):
    """Stores files in AWS S3. For production deployment."""

    def __init__(self):
        import boto3
        settings = get_settings()
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        self.bucket = settings.s3_bucket

    def store_raw_match(self, match_id: int, data: dict, source: str = "opendota") -> str:
        key = f"raw/{source}/{match_id}.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=orjson.dumps(data),
            ContentType="application/json",
        )
        return key

    def get_raw_match(self, match_id: int, source: str = "opendota") -> dict | None:
        key = f"raw/{source}/{match_id}.json"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return orjson.loads(response["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None

    def store_model(self, model_name: str, version: str, model_bytes: bytes) -> str:
        key = f"models/{model_name}/{version}.pkl"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=model_bytes,
            ContentType="application/octet-stream",
        )
        return key

    def get_model(self, model_name: str, version: str) -> bytes | None:
        key = f"models/{model_name}/{version}.pkl"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def list_model_versions(self, model_name: str) -> list[str]:
        prefix = f"models/{model_name}/"
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [
            obj["Key"].split("/")[-1].replace(".pkl", "")
            for obj in response["Contents"]
        ]


# ── Factory ────────────────────────────────────────────────

_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """
    Returns the configured storage backend (singleton).
    Set STORAGE_BACKEND=s3 in .env to use S3, otherwise uses local disk.
    """
    global _instance
    if _instance is None:
        settings = get_settings()
        if settings.storage_backend == "s3":
            logger.info("Using S3 storage backend")
            _instance = S3StorageBackend()
        else:
            logger.info(f"Using local storage backend at {settings.storage_local_dir}")
            _instance = LocalStorageBackend(settings.storage_local_dir)
    return _instance
