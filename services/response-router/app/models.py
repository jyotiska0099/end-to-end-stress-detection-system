from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Patient(Base):
    __tablename__ = "patients"

    patient_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    latest_label: Mapped[int] = mapped_column(Integer, nullable=False)
    latest_severity: Mapped[str] = mapped_column(String(16), nullable=False)
    latest_probability: Mapped[float] = mapped_column(Float, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class StressEvent(Base):
    __tablename__ = "stress_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    stress_probability: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    urgency_level: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    follow_up_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
