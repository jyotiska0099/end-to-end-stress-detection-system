from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    patient_id: str
    stress_probability: float = Field(..., ge=0.0, le=1.0)
    label: int = Field(..., ge=0, le=1)


class StressRecommendation(BaseModel):
    severity: str  # "none" | "mild" | "moderate" | "high"
    urgency_level: int  # 1–5
    summary: str  # one sentence
    recommendation: str  # actionable advice
    follow_up_minutes: int  # suggested check-in interval


class LLMResponse(BaseModel):
    patient_id: str
    recommendation: StressRecommendation
    raw_prompt_tokens: int
    raw_output_tokens: int
