from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Auth
    ingest_api_key: str = "changeme-32-byte-secret-here"
    hmac_secret: str = "changeme-hmac-secret"  # stub — activated in Phase 7
    hmac_enabled: bool = False  # flip to True when ready

    # Upstream services
    inference_service_url: str = "http://localhost:8000"
    llm_service_url: str = "http://localhost:8001"

    # Database (SQLite for local dev, swapped to PostgreSQL in Phase 7)
    database_url: str = "sqlite+aiosqlite:///./stress.db"

    response_router_port: int = 8002
    log_level: str = "INFO"


settings = Settings()
