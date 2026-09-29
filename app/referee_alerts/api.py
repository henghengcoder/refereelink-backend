"""REST and WebSocket surface for referee alerts.

Mounted only on the main server app (``app/server/main.py``); the standalone
SRT receiver app must not expose the alert downlink.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from app.field_ingest.service import FieldIngestService
from app.referee_alerts.models import (
    CreateRefereeAlertRequest,
    RefereeAlert,
    RefereeAlertSource,
)
from app.referee_alerts.service import RefereeAlertService

router = APIRouter(tags=["referee-alerts"])
logger = logging.getLogger(__name__)


def _service(request: Request) -> RefereeAlertService:
    return request.app.state.referee_alerts  # type: ignore[attr-defined]


def _bearer(value: str | None) -> str | None:
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def _field_authorized(websocket: WebSocket) -> bool:
    """The downlink reuses the field-ingest bearer token boundary, exactly like
    the sibling ``/ws/v1/field/*`` routes: unconfigured or mismatched tokens are
    rejected before ``accept()``."""
    field_ingest: FieldIngestService | None = getattr(websocket.app.state, "field_ingest", None)
    if field_ingest is None or not field_ingest.settings.auth_token:
        return False
    return field_ingest.authorize(_bearer(websocket.headers.get("authorization")))


@router.post("/api/referee-alerts", response_model=RefereeAlert)
async def create_referee_alert(
    payload: CreateRefereeAlertRequest, request: Request
) -> RefereeAlert:
    """Create a manual referee alert and publish it once.

    The server owns identity and provenance: ``event_id``/``timestamp`` are
    generated here, ``source`` is always ``manual``, and ``evidence``/
    ``case_id`` stay ``null`` until detector/review integration exists.
    """
    alert = RefereeAlert(
        event_id=str(uuid.uuid4()),
        type=payload.type,
        timestamp=time.time(),
        confidence=payload.confidence,
        evidence=None,
        case_id=None,
        source=RefereeAlertSource.MANUAL,
    )
    delivered = await _service(request).publish_referee_alert(alert)
    if not delivered:  # defensive: each request mints a fresh uuid4
        logger.warning("Manual referee alert %s was deduplicated", alert.event_id)
    return alert


@router.websocket("/ws/v1/field/alerts")
async def alerts_socket(websocket: WebSocket) -> None:
    """Field downlink: pushes raw ``RefereeAlert`` JSON frames and records
    ``{"type": "acknowledged", "event_id": ...}`` confirmations in memory."""
    service: RefereeAlertService = websocket.app.state.referee_alerts  # type: ignore[attr-defined]
    if not _field_authorized(websocket):
        await websocket.close(code=1008, reason="invalid field ingest bearer token")
        return
    await service.connect(websocket)
    try:
        while True:
            try:
                message: Any = await websocket.receive_json()
            except ValueError:  # malformed JSON: answer and keep the session
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "INVALID_MESSAGE",
                        "detail": "frames must be JSON objects",
                    }
                )
                continue
            if service.handle_client_message(message):
                continue
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "INVALID_ACK",
                    "detail": "expected {'type': 'acknowledged', 'event_id': '<event_id>'}",
                }
            )
    except WebSocketDisconnect:
        return
    finally:
        await service.disconnect(websocket)
