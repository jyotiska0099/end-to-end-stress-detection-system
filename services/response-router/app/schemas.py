from datetime import datetime

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    patient_id: str
    eda: list[float] = Field(..., min_length=240, max_length=240)
    bvp: list[float] = Field(..., min_length=3840, max_length=3840)


class IngestResponse(BaseModel):
    patient_id: str
    label: int
    severity: str
    urgency_level: int
    summary: str
    recommendation: str
    follow_up_minutes: int
    event_id: int


class PatientStatus(BaseModel):
    patient_id: str
    latest_label: int
    latest_severity: str
    latest_probability: float
    last_seen_at: datetime


class WSMessage(BaseModel):
    event: str = "stress_update"
    patient_id: str
    label: int
    severity: str
    urgency_level: int
    summary: str
    recommendation: str
    follow_up_minutes: int
    stress_probability: float
    event_id: int
