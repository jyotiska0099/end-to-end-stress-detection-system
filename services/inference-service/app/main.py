import logging
import time

from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from app import model as model_store
from app.config import settings
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stress Inference Service", version="1.0.0")

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("inference_requests_total", "Total prediction requests", ["label"])
REQUEST_LATENCY = Histogram("inference_latency_seconds", "Prediction latency")


@app.on_event("startup")
async def startup():
    try:
        model_store.load_model()
    except RuntimeError as e:
        logger.warning("Model not loaded on startup: %s", e)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_store.is_loaded(),
        device=str(model_store._device),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model_store.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.perf_counter()
    try:
        prob, label = model_store.predict(request.eda, request.bvp)
    except Exception as e:
        logger.exception("Inference error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    latency = time.perf_counter() - t0
    REQUEST_LATENCY.observe(latency)
    REQUEST_COUNT.labels(label=str(label)).inc()

    logger.info(
        "patient=%s  prob=%.3f  label=%d  latency=%.3fs", request.patient_id, prob, label, latency
    )

    return PredictResponse(
        patient_id=request.patient_id,
        stress_probability=round(prob, 4),
        label=label,
        model_run_id=model_store._run_id,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
