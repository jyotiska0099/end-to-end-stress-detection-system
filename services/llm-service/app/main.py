import logging

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app.config import settings
from app.gemini import get_recommendation
from app.schemas import LLMRequest, LLMResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Recommendation Service", version="1.0.0")

REQUEST_COUNT = Counter("llm_requests_total", "Total LLM requests", ["severity"])
REQUEST_LATENCY = Histogram("llm_latency_seconds", "Gemini call latency")
TOKEN_COUNTER = Counter("llm_tokens_total", "Total tokens used", ["type"])


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.gemini_model}


@app.post("/recommend", response_model=LLMResponse)
def recommend(request: LLMRequest):
    try:
        with REQUEST_LATENCY.time():
            rec, prompt_tokens, output_tokens = get_recommendation(
                patient_id=request.patient_id,
                stress_probability=request.stress_probability,
                label=request.label,
            )
    except Exception as e:
        logger.exception("Gemini call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    REQUEST_COUNT.labels(severity=rec.severity).inc()
    TOKEN_COUNTER.labels(type="prompt").inc(prompt_tokens)
    TOKEN_COUNTER.labels(type="output").inc(output_tokens)

    logger.info(
        "patient=%s severity=%s urgency=%d follow_up=%dmin",
        request.patient_id,
        rec.severity,
        rec.urgency_level,
        rec.follow_up_minutes,
    )

    return LLMResponse(
        patient_id=request.patient_id,
        recommendation=rec,
        raw_prompt_tokens=prompt_tokens,
        raw_output_tokens=output_tokens,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
