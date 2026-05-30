from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str
    # Change model as gemini-2.0-flash-lite is depricated from June 1"
    gemini_model: str = "gemini-2.0-flash-lite"
    llm_service_port: int = 8001
    log_level: str = "INFO"


settings = Settings()
