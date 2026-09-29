"""Publishing hub for referee alerts.

``RefereeAlertService.publish_referee_alert`` is the single entry point that
future detectors and the manual REST route share. It:

- deduplicates by ``event_id`` (first delivery wins; duplicates are idempotent
  no-ops and are neither queued, mirrored, nor pushed again);
- appends the alert to a bounded in-memory queue;
- mirrors the alert into the existing dashboard event pipeline as a
  ``GameEvent`` (``StateStore`` for ``GET /api/events`` plus one
  ``referee_alert`` broadcast on the shared ``/ws/state`` publisher);
- pushes the raw ``RefereeAlert`` JSON (no envelope) to every session
  connected on ``/ws/v1/field/alerts``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Any

from fastapi import WebSocket

from app.referee_alerts.models import (
    RefereeAlert,
    RefereeAlertAcknowledgement,
)
from app.services.publisher import WebSocketPublisher
from app.state.models import FrameState, GameEvent
from app.state.store import StateStore

logger = logging.getLogger(__name__)

MAX_ALERTS = 500
MAX_ACKNOWLEDGEMENTS = 500
# Manual/detector alerts carry no frame of their own; when no pipeline frame
# has been processed yet the mirrored GameEvent reports frame 0.
NO_FRAME_ID = 0
DASHBOARD_BROADCAST_TYPE = "referee_alert"


class RefereeAlertService:
    def __init__(
        self,
        store: StateStore,
        publisher: WebSocketPublisher | None = None,
        *,
        max_alerts: int = MAX_ALERTS,
        max_acknowledgements: int = MAX_ACKNOWLEDGEMENTS,
    ) -> None:
        self._store = store
        self._publisher = publisher
        self._lock = threading.Lock()
        self._published_event_ids: set[str] = set()
        self._alerts: deque[RefereeAlert] = deque(maxlen=max_alerts)
        self._acknowledgements: deque[RefereeAlertAcknowledgement] = deque(
            maxlen=max_acknowledgements
        )
        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()

    @property
    def alerts(self) -> list[RefereeAlert]:
        with self._lock:
            return list(self._alerts)

    @property
    def acknowledgements(self) -> list[RefereeAlertAcknowledgement]:
        with self._lock:
            return list(self._acknowledgements)

    @property
    def client_count(self) -> int:
        # Diagnostic count; len() on a set is atomic, no async lock needed.
        return len(self._clients)

    async def publish_referee_alert(self, alert: RefereeAlert) -> bool:
        """Publish ``alert`` exactly once.

        Returns ``True`` when the alert was delivered for the first time and
        ``False`` when ``event_id`` was already published (idempotent no-op).
        """
        with self._lock:
            if alert.event_id in self._published_event_ids:
                return False
            self._published_event_ids.add(alert.event_id)
            self._alerts.append(alert)
            game_event = self._as_game_event(alert)
            self._store.add_event(game_event)
        if self._publisher is not None:
            await self._publisher.broadcast(
                {
                    "type": DASHBOARD_BROADCAST_TYPE,
                    "event": game_event.model_dump(mode="json"),
                }
            )
        await self._push_to_clients(alert.model_dump(mode="json"))
        logger.info(
            "Referee alert %s (%s, %s) published to %d downlink session(s)",
            alert.event_id,
            alert.type.value,
            alert.source.value,
            self.client_count,
        )
        return True

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._clients_lock:
            self._clients.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._clients_lock:
            self._clients.discard(websocket)

    def handle_client_message(self, data: Any) -> bool:
        """Record one downstream frame; ``True`` when it is a valid ack."""
        if not isinstance(data, dict) or data.get("type") != "acknowledged":
            return False
        event_id = data.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            return False
        with self._lock:
            self._acknowledgements.append(RefereeAlertAcknowledgement(event_id=event_id))
        return True

    def _as_game_event(self, alert: RefereeAlert) -> GameEvent:
        """Map the alert onto the dashboard's existing GameEvent pipeline."""
        frame_state: FrameState | None = self._store.latest_frame_state
        return GameEvent(
            id=alert.event_id,
            event_type=alert.type.value,
            confidence=alert.confidence,
            severity="candidate",
            timestamp=alert.timestamp,
            frame_id=frame_state.frame_id if frame_state is not None else NO_FRAME_ID,
            evidence=alert.evidence if alert.evidence is not None else {},
        )

    async def _push_to_clients(self, payload: dict[str, Any]) -> None:
        async with self._clients_lock:
            dead: list[WebSocket] = []
            for websocket in self._clients:
                try:
                    await websocket.send_json(payload)
                except Exception:
                    dead.append(websocket)
            for websocket in dead:
                self._clients.discard(websocket)
