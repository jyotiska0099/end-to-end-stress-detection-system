import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = str(Path(os.getcwd()) / "mlruns")

    # MLflow (local for now — switches to remote URI in Phase 7)
    mlflow_run_id: str = ""  # populated after first training run
    model_artifact_name: str = "best_model_S17.pt"
    eda_scaler_artifact: str = "eda_scaler_S17.pkl"
    bvp_scaler_artifact: str = "bvp_scaler_S17.pkl"

    # Service
    inference_service_port: int = 8000
    llm_service_url: str = "http://llm-service:8001"
    log_level: str = "INFO"


settings = Settings()
