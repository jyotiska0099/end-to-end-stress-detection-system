from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    patient_id: str
    eda: list[float] = Field(
        ..., min_length=240, max_length=240, description="60s EDA at 4 Hz → 240 samples"
    )
    bvp: list[float] = Field(
        ..., min_length=3840, max_length=3840, description="60s BVP at 64 Hz → 3840 samples"
    )


class PredictResponse(BaseModel):
    patient_id: str
    stress_probability: float
    label: int  # 0 = no_stress, 1 = stress
    model_run_id: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
