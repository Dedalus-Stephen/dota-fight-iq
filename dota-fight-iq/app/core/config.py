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
    storage_backend: str = "local"  # "local" or "s3"
    storage_local_dir: str = "./data"  # Used when storage_backend = "local"

    # AWS (only required when storage_backend = "s3" or deploying to AWS)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "eu-west-1"
    s3_bucket: str = "dota-fight-iq-data"

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
