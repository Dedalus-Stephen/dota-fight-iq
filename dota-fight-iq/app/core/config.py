from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    env: str = "development"
    debug: bool = True

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    database_url: str

    # Storage
    storage_backend: str = "local"  # "local" or "gcs"
    storage_local_dir: str = "./data"  # Used when storage_backend = "local"

    # GCP (only required when storage_backend = "gcs" or deploying to Cloud Run)
    gcp_project_id: str = ""
    gcs_bucket: str = "dota-fight-iq-data"
    # Authentication: uses Application Default Credentials (ADC).
    # - Local dev: run `gcloud auth application-default login`
    # - Cloud Run: automatic via attached service account
    # - CI/CD: set GOOGLE_APPLICATION_CREDENTIALS env var pointing to a key file
    gcp_credentials_path: str = ""  # Optional: explicit path to service account JSON

    # External APIs
    opendota_api_key: str = ""  # Optional, increases rate limit
    stratz_api_token: str = ""  # Optional for basic tier

    # Rate limits
    opendota_rate_limit: int = 60  # calls per minute
    stratz_rate_limit: int = 10000  # calls per day

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()