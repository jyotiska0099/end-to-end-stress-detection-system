import logging

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from app import ws_manager
from app.config import settings
from app.database import get_db, init_db
from app.models import Patient
from app.orchestrator import process_ingest
from app.schemas import IngestRequest, IngestResponse, PatientStatus
from app.security import verify_api_key

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="Response Router", version="1.0.0")

INGEST_COUNT = Counter("router_ingest_total", "Total ingest requests", ["severity"])
INGEST_LATENCY = Histogram("router_ingest_latency_seconds", "End-to-end pipeline latency")


@app.on_event("startup")
async def startup():
    await init_db()
    logger.info("Database initialised.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
async def ingest(request: IngestRequest, db: AsyncSession = Depends(get_db)):
    with INGEST_LATENCY.time():
        result = await process_ingest(request, db)
    INGEST_COUNT.labels(severity=result.severity).inc()
    return result


@app.get("/patients", response_model=list[PatientStatus])
async def list_patients(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient))
    return result.scalars().all()


@app.get("/patients/{patient_id}", response_model=PatientStatus)
async def get_patient(patient_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = result.scalar_one_or_none()
    if patient is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(patient_id: str, ws: WebSocket):
    await ws_manager.subscribe(patient_id, ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive; client can send pings
    except WebSocketDisconnect:
        ws_manager.unsubscribe(patient_id, ws)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
