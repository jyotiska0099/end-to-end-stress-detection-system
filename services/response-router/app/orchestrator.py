"""
Pipeline orchestrator: inference → LLM → DB → WebSocket fan-out.
"""

import logging

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import ws_manager
from app.config import settings
from app.models import Patient, StressEvent
from app.schemas import IngestRequest, IngestResponse, WSMessage

logger = logging.getLogger(__name__)


async def process_ingest(request: IngestRequest, db: AsyncSession) -> IngestResponse:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── Step 1: Inference ──────────────────────────────────────────────────
        infer_resp = await client.post(
            f"{settings.inference_service_url}/predict",
            json={
                "patient_id": request.patient_id,
                "eda": request.eda,
                "bvp": request.bvp,
            },
        )
        infer_resp.raise_for_status()
        infer = infer_resp.json()

        # ── Step 2: LLM recommendation ────────────────────────────────────────
        llm_resp = await client.post(
            f"{settings.llm_service_url}/recommend",
            json={
                "patient_id": request.patient_id,
                "stress_probability": infer["stress_probability"],
                "label": infer["label"],
            },
        )
        llm_resp.raise_for_status()
        llm = llm_resp.json()
        rec = llm["recommendation"]

        # ── Step 3: Upsert patients table ─────────────────────────────────────
        result = await db.execute(select(Patient).where(Patient.patient_id == request.patient_id))
        patient = result.scalar_one_or_none()
        if patient is None:
            patient = Patient(
                patient_id=request.patient_id,
                latest_label=infer["label"],
                latest_severity=rec["severity"],
                latest_probability=infer["stress_probability"],
            )
            db.add(patient)
        else:
            patient.latest_label = infer["label"]
            patient.latest_severity = rec["severity"]
            patient.latest_probability = infer["stress_probability"]

        # ── Step 4: Insert stress_events ──────────────────────────────────────
        event = StressEvent(
            patient_id=request.patient_id,
            stress_probability=infer["stress_probability"],
            label=infer["label"],
            severity=rec["severity"],
            urgency_level=rec["urgency_level"],
            summary=rec["summary"],
            recommendation=rec["recommendation"],
            follow_up_minutes=rec["follow_up_minutes"],
        )
        db.add(event)
        await db.commit()
        await db.refresh(event)

        # ── Step 5: WebSocket fan-out ─────────────────────────────────────────
        ws_msg = WSMessage(
            patient_id=request.patient_id,
            label=infer["label"],
            severity=rec["severity"],
            urgency_level=rec["urgency_level"],
            summary=rec["summary"],
            recommendation=rec["recommendation"],
            follow_up_minutes=rec["follow_up_minutes"],
            stress_probability=infer["stress_probability"],
            event_id=event.id,
        )
        await ws_manager.broadcast(request.patient_id, ws_msg.model_dump())

        logger.info(
            "Processed: patient=%s label=%d severity=%s urgency=%d",
            request.patient_id,
            infer["label"],
            rec["severity"],
            rec["urgency_level"],
        )

        return IngestResponse(
            patient_id=request.patient_id,
            label=infer["label"],
            severity=rec["severity"],
            urgency_level=rec["urgency_level"],
            summary=rec["summary"],
            recommendation=rec["recommendation"],
            follow_up_minutes=rec["follow_up_minutes"],
            event_id=event.id,
        )
