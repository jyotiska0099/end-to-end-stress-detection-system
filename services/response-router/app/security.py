"""
Security middleware for the response router.

Active now:   Shared secret header (X-Api-Key)
Stub ready:   HMAC-SHA256 payload signature (activate via HMAC_ENABLED=true)
Production:   mTLS via Istio (Phase 8, zero app-level code needed)
"""

import hashlib
import hmac
import logging

from fastapi import Header, HTTPException, Request

from app.config import settings

logger = logging.getLogger(__name__)


# ── Shared secret header (active) ─────────────────────────────────────────────


def verify_api_key(x_api_key: str = Header(..., alias="X-Api-Key")) -> None:
    """FastAPI dependency — validates X-Api-Key header."""
    if x_api_key != settings.ingest_api_key:
        logger.warning("Rejected request: invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ── HMAC-SHA256 payload signature (stub — not active) ─────────────────────────


async def verify_hmac_signature(request: Request) -> None:
    """
    FastAPI dependency stub for HMAC-SHA256 payload verification.

    To activate:
      1. Set HMAC_ENABLED=true in .env
      2. Add this as a dependency alongside verify_api_key
      3. Simulator must set header:
           X-Signature: sha256=<hmac_hex>
         where hmac_hex = HMAC-SHA256(secret, raw_request_body).hexdigest()

    This is a defence-in-depth measure — verifies the payload was not
    tampered with in transit, even if the API key was compromised.
    """
    if not settings.hmac_enabled:
        return  # stub: passthrough until activated

    signature_header = request.headers.get("X-Signature", "")
    if not signature_header.startswith("sha256="):
        raise HTTPException(status_code=401, detail="Missing HMAC signature.")

    provided_sig = signature_header[len("sha256=") :]
    body = await request.body()
    expected_sig = hmac.new(
        settings.hmac_secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(provided_sig, expected_sig):
        logger.warning("Rejected request: HMAC signature mismatch")
        raise HTTPException(status_code=401, detail="Invalid payload signature.")
