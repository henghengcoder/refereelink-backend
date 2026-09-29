"""Referee alert publishing and Field downlink.

``RefereeAlert`` is the wire contract shared by the backend, the web dashboard
and the iOS Field/watch apps. ``RefereeAlertService.publish_referee_alert`` is
the single delivery point that the manual ``POST /api/referee-alerts`` route
uses today and future detectors will call. Alerts are deduplicated by
``event_id``, mirrored into the dashboard event pipeline (``StateStore`` plus
one ``referee_alert`` broadcast on the shared ``/ws/state`` publisher), and
pushed verbatim (no envelope) to every session on ``/ws/v1/field/alerts``.
"""

from app.referee_alerts.api import router as referee_alert_router
from app.referee_alerts.models import (
    CreateRefereeAlertRequest,
    RefereeAlert,
    RefereeAlertAcknowledgement,
    RefereeAlertSource,
    RefereeAlertType,
)
from app.referee_alerts.service import RefereeAlertService

__all__ = [
    "CreateRefereeAlertRequest",
    "RefereeAlert",
    "RefereeAlertAcknowledgement",
    "RefereeAlertService",
    "RefereeAlertSource",
    "RefereeAlertType",
    "referee_alert_router",
]
