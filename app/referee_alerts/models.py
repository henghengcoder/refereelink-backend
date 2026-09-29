"""Wire models for the referee alert channel.

``RefereeAlert`` is the shared contract across the backend, the web dashboard
and the iOS Field/watch clients:

- it is the JSON body created by ``POST /api/referee-alerts`` and returned as
  the response of that call;
- it is the downstream WebSocket frame pushed on ``/ws/v1/field/alerts``
  verbatim (no envelope), so Field clients parse the frame as-is;
- ``case_id`` stays ``null`` until the multi-view review hop is wired up, and
  no client may open a review page while it is ``null``.

Field clients confirm delivery with ``{"type": "acknowledged",
"event_id": ...}``; the server records those frames in memory only.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RefereeAlertType(str, Enum):
    FOUL_CANDIDATE = "foul_candidate"
    OFFSIDE_CANDIDATE = "offside_candidate"


class RefereeAlertSource(str, Enum):
    MANUAL = "manual"
    DETECTOR = "detector"


class RefereeAlert(BaseModel):
    """The alert payload itself; doubles as REST response and WS frame."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: RefereeAlertType
    timestamp: float = Field(default_factory=time.time)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: Optional[dict[str, Any]] = None
    case_id: Optional[str] = None
    source: RefereeAlertSource


class CreateRefereeAlertRequest(BaseModel):
    """Manual alert request body; the server owns every other field."""

    type: RefereeAlertType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RefereeAlertAcknowledgement(BaseModel):
    """In-memory record of one ``acknowledged`` frame from a client."""

    event_id: str
    received_at: float = Field(default_factory=time.time)
