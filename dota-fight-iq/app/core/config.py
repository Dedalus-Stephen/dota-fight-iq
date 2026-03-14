from pydantic_settings import BaseSettings, SettingsConfigDict
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
    storage_local_dir: str = "./data"

    # GCP
    gcp_project_id: str = ""
    gcs_bucket: str = "dota-fight-iq-data"
    gcp_credentials_path: str = ""

    # External APIs
    opendota_api_key: str = ""
    stratz_api_token: str = ""

    # Rate limits
    opendota_rate_limit: int = 60
    stratz_rate_limit: int = 10000 
    
    # GCP Specific (The ones causing the crash)
    parser_worker_url: str
    cloud_run_sa_email: str
    gcp_region: str = "us-central1"
    
    # Steam GC Retriever
    steam_user: str = ""
    steam_pass: str = ""

    # --- MERGED CONFIG ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # This is the "shield" that prevents the crash
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()