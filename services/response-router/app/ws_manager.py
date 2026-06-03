"""
WebSocket connection manager.

Each patient_id maps to a set of connected WebSocket clients.
When a new stress event arrives, all subscribers for that patient are notified.
Designed for multi-doctor expansion: each doctor connects to ws/{patient_id}
for each patient they want to monitor.
"""

import logging
from collections import defaultdict

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# patient_id → set of active WebSocket connections
_subscriptions: dict[str, set[WebSocket]] = defaultdict(set)


async def subscribe(patient_id: str, ws: WebSocket) -> None:
    await ws.accept()
    _subscriptions[patient_id].add(ws)
    logger.info("WS subscribed: patient=%s  total=%d", patient_id, len(_subscriptions[patient_id]))


def unsubscribe(patient_id: str, ws: WebSocket) -> None:
    _subscriptions[patient_id].discard(ws)
    logger.info(
        "WS unsubscribed: patient=%s  remaining=%d", patient_id, len(_subscriptions[patient_id])
    )


async def broadcast(patient_id: str, message: dict) -> None:
    """Fan-out message to all subscribers of a patient."""
    dead: set[WebSocket] = set()
    for ws in _subscriptions.get(patient_id, set()):
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)

    # Clean up disconnected clients
    for ws in dead:
        _subscriptions[patient_id].discard(ws)
