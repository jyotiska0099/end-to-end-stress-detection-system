"""
Sensor Simulator — replays WESAD wrist signals sequentially for N patients,
posting each 60s window to the response router at a fixed interval.

Usage:
    python simulator.py                          # default: S17, every 5s
    python simulator.py --subjects S15 S16 S17  # multi-patient
    python simulator.py --interval 2             # faster replay
    python simulator.py --data-dir /path/to/wesad
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# Make ml package importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.preprocessor import window_subject
from ml.data.wesad_loader import load_subject

# ── Config from env ────────────────────────────────────────────────────────────
ROUTER_URL = os.getenv("INFERENCE_URL", "http://localhost:8002/ingest")
API_KEY = os.getenv("INGEST_API_KEY", "changeme-32-byte-secret-here")
INTERVAL = int(os.getenv("SIGNAL_INTERVAL_SECONDS", "5"))
DATA_DIR = Path(os.getenv("WESAD_DATA_DIR", "ml/data/wesad"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("simulator")


async def replay_subject(
    subject: str,
    windows: list[dict],
    interval: float,
    client: httpx.AsyncClient,
) -> None:
    """
    Sequentially POST each window for a subject to the router,
    waiting `interval` seconds between each.
    Loops back to the start when all windows are exhausted.
    """
    logger.info(
        "Starting replay for %s — %d windows, interval=%ss", subject, len(windows), interval
    )
    idx = 0
    while True:
        window = windows[idx % len(windows)]
        idx += 1

        payload = {
            "patient_id": subject,
            "eda": window["EDA"].tolist(),
            "bvp": window["BVP"].tolist(),
        }

        try:
            resp = await client.post(
                ROUTER_URL,
                json=payload,
                headers={"X-Api-Key": API_KEY},
                timeout=30.0,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "patient=%-4s  window=%03d  label=%d  severity=%-8s  urgency=%d  prob=%.3f",
                subject,
                idx,
                result["label"],
                result["severity"],
                result["urgency_level"],
                # prob not in IngestResponse — log what we have
                0.0,
            )
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error for %s: %s", subject, e)
        except httpx.RequestError as e:
            logger.error("Connection error for %s: %s — retrying next cycle", subject, e)

        await asyncio.sleep(interval)


async def main(subjects: list[str], interval: float, data_dir: Path) -> None:
    logger.info("Loading WESAD windows for subjects: %s", subjects)

    subject_windows: dict[str, list[dict]] = {}
    for s in subjects:
        try:
            raw = load_subject(data_dir, s)
            windows = window_subject(raw)
            if not windows:
                logger.warning("No windows generated for %s — skipping", s)
                continue
            subject_windows[s] = windows
            logger.info("  %s: %d windows loaded", s, len(windows))
        except FileNotFoundError:
            logger.error("WESAD data not found for %s at %s", s, data_dir)

    if not subject_windows:
        logger.error("No subjects loaded. Exiting.")
        return

    async with httpx.AsyncClient() as client:
        # Run all subjects concurrently, staggered by 1s to avoid burst
        tasks = []
        for i, (subject, windows) in enumerate(subject_windows.items()):
            await asyncio.sleep(1)
            tasks.append(asyncio.create_task(replay_subject(subject, windows, interval, client)))
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WESAD sensor simulator")
    parser.add_argument(
        "--subjects", nargs="+", default=["S17"], help="Subject IDs to simulate (default: S17)"
    )
    parser.add_argument(
        "--interval", type=float, default=INTERVAL, help="Seconds between window posts (default: 5)"
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Path to WESAD root directory")
    args = parser.parse_args()

    asyncio.run(
        main(
            subjects=args.subjects,
            interval=args.interval,
            data_dir=Path(args.data_dir),
        )
    )
