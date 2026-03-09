"""
Storage Abstraction Layer

Provides a unified interface for storing raw match JSONs and model artifacts.
Backend is selected by STORAGE_BACKEND env var:
  - "local" (default): stores files on local disk
  - "gcs": stores files in Google Cloud Storage

Switching from local to GCS requires only changing STORAGE_BACKEND and
configuring GCP credentials (ADC or service account). No code changes
anywhere else in the application.
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


# ── Google Cloud Storage Backend ──────────────────────────

class GCSStorageBackend(StorageBackend):
    """Stores files in Google Cloud Storage. For production deployment.

    Authentication uses Application Default Credentials (ADC):
    - Local dev: `gcloud auth application-default login`
    - Cloud Run: automatic via attached service account
    - CI/CD: set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path
    """

    def __init__(self):
        from google.cloud import storage as gcs_storage

        settings = get_settings()

        # If an explicit credentials path is set, use it.
        # Otherwise, rely on ADC (which Cloud Run provides automatically).
        if settings.gcp_credentials_path:
            self.client = gcs_storage.Client.from_service_account_json(
                settings.gcp_credentials_path,
                project=settings.gcp_project_id or None,
            )
        else:
            self.client = gcs_storage.Client(
                project=settings.gcp_project_id or None,
            )

        self.bucket_name = settings.gcs_bucket
        self.bucket = self.client.bucket(self.bucket_name)
        logger.info(f"GCS storage initialized: gs://{self.bucket_name}")

    def store_raw_match(self, match_id: int, data: dict, source: str = "opendota") -> str:
        key = f"raw/{source}/{match_id}.json"
        blob = self.bucket.blob(key)
        blob.upload_from_string(
            orjson.dumps(data),
            content_type="application/json",
        )
        logger.debug(f"Stored raw match {match_id} at gs://{self.bucket_name}/{key}")
        return key

    def get_raw_match(self, match_id: int, source: str = "opendota") -> dict | None:
        key = f"raw/{source}/{match_id}.json"
        blob = self.bucket.blob(key)
        if not blob.exists():
            return None
        return orjson.loads(blob.download_as_bytes())

    def store_model(self, model_name: str, version: str, model_bytes: bytes) -> str:
        key = f"models/{model_name}/{version}.pkl"
        blob = self.bucket.blob(key)
        blob.upload_from_string(
            model_bytes,
            content_type="application/octet-stream",
        )
        logger.debug(f"Stored model {model_name}/{version} at gs://{self.bucket_name}/{key}")
        return key

    def get_model(self, model_name: str, version: str) -> bytes | None:
        key = f"models/{model_name}/{version}.pkl"
        blob = self.bucket.blob(key)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def list_model_versions(self, model_name: str) -> list[str]:
        prefix = f"models/{model_name}/"
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [
            blob.name.split("/")[-1].replace(".pkl", "")
            for blob in blobs
            if blob.name.endswith(".pkl")
        ]


# ── Factory ────────────────────────────────────────────────

_instance: StorageBackend | None = None


def get_storage() -> StorageBackend:
    """
    Returns the configured storage backend (singleton).
    Set STORAGE_BACKEND=gcs in .env to use Google Cloud Storage,
    otherwise uses local disk.
    """
    global _instance
    if _instance is None:
        settings = get_settings()
        if settings.storage_backend == "gcs":
            logger.info("Using GCS storage backend")
            _instance = GCSStorageBackend()
        else:
            logger.info(f"Using local storage backend at {settings.storage_local_dir}")
            _instance = LocalStorageBackend(settings.storage_local_dir)
    return _instance