from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.field_ingest.config import FieldIngestSettings
from app.field_ingest.service import FieldIngestService
from app.referee_alerts.api import router as referee_alert_router
from app.referee_alerts.models import (
    RefereeAlert,
    RefereeAlertSource,
    RefereeAlertType,
)
from app.referee_alerts.service import RefereeAlertService
from app.server.main import app
from app.services.publisher import WebSocketPublisher
from app.state.models import FrameState
from app.state.store import StateStore

CONTRACT_FIELDS = {"event_id", "type", "timestamp", "confidence", "evidence", "case_id", "source"}

FIELD_TEST_TOKEN = "field-test-token"


class _FakeDownlinkClient:
    """Duck-typed WebSocket for WebSocketPublisher.connect()."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, message: dict) -> None:
        self.sent.append(message)


def _alert(**overrides) -> RefereeAlert:
    payload: dict = {
        "type": RefereeAlertType.FOUL_CANDIDATE,
        "source": RefereeAlertSource.DETECTOR,
        "confidence": 0.75,
    }
    payload.update(overrides)
    return RefereeAlert(**payload)


@pytest.fixture
def alert_client():
    """TestClient with an isolated StateStore and RefereeAlertService so the
    process-wide singletons in app.server.main stay untouched."""
    previous_store = app.state.store
    previous_service = app.state.referee_alerts
    fresh_store = StateStore()
    app.state.store = fresh_store
    app.state.referee_alerts = RefereeAlertService(fresh_store, app.state.publisher)
    try:
        yield TestClient(app)
    finally:
        app.state.store = previous_store
        app.state.referee_alerts = previous_service


async def test_publish_referee_alert_is_idempotent_per_event_id():
    store = StateStore()
    service = RefereeAlertService(store)
    alert = _alert()

    assert await service.publish_referee_alert(alert) is True
    assert await service.publish_referee_alert(alert) is False

    assert [a.event_id for a in service.alerts] == [alert.event_id]
    assert [event.id for event in store.events] == [alert.event_id]
    # No pipeline frame has been processed, so the mirrored event reports 0.
    assert store.events[0].frame_id == 0


async def test_duplicate_publish_does_not_push_or_mirror_twice():
    store = StateStore()
    publisher = WebSocketPublisher(store)
    downlink = _FakeDownlinkClient()
    await publisher.connect(downlink)
    service = RefereeAlertService(store, publisher)
    alert = _alert()

    assert await service.publish_referee_alert(alert) is True
    assert await service.publish_referee_alert(alert) is False

    assert len(downlink.sent) == 1
    assert len(service.alerts) == 1
    assert len(store.events) == 1


async def test_publish_mirrors_alert_into_dashboard_event_pipeline():
    store = StateStore()
    store.latest_frame_state = FrameState(frame_id=77)
    publisher = WebSocketPublisher(store)
    dashboard_client = _FakeDownlinkClient()
    await publisher.connect(dashboard_client)
    service = RefereeAlertService(store, publisher)
    alert = _alert(evidence={"ball_x": 12.5}, timestamp=123.5, confidence=0.4)

    assert await service.publish_referee_alert(alert) is True

    event = store.events[-1]
    assert event.id == alert.event_id
    assert event.event_type == "foul_candidate"
    assert event.confidence == 0.4
    assert event.severity == "candidate"
    assert event.timestamp == 123.5
    assert event.frame_id == 77
    assert event.evidence == {"ball_x": 12.5}
    assert dashboard_client.sent == [
        {"type": "referee_alert", "event": event.model_dump(mode="json")}
    ]


def test_referee_alert_wire_contract_serializes_exact_fields():
    alert = RefereeAlert(
        event_id="e11",
        type=RefereeAlertType.FOUL_CANDIDATE,
        timestamp=12.5,
        confidence=0.9,
        evidence={"zone": "box"},
        case_id=None,
        source=RefereeAlertSource.DETECTOR,
    )
    payload = json.loads(alert.model_dump_json())
    assert payload == {
        "event_id": "e11",
        "type": "foul_candidate",
        "timestamp": 12.5,
        "confidence": 0.9,
        "evidence": {"zone": "box"},
        "case_id": None,
        "source": "detector",
    }


def test_post_referee_alert_creates_manual_alert_with_server_owned_fields(alert_client):
    started = time.time()
    response = alert_client.post("/api/referee-alerts", json={"type": "foul_candidate"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == CONTRACT_FIELDS
    assert body["type"] == "foul_candidate"
    assert body["source"] == "manual"
    assert body["confidence"] == 1.0
    assert body["evidence"] is None
    assert body["case_id"] is None
    uuid.UUID(body["event_id"])  # server-generated UUID string
    assert isinstance(body["timestamp"], float)
    assert started - 1.0 <= body["timestamp"] <= time.time() + 1.0


def test_post_referee_alert_accepts_optional_confidence(alert_client):
    response = alert_client.post(
        "/api/referee-alerts", json={"type": "offside_candidate", "confidence": 0.6}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "offside_candidate"
    assert body["confidence"] == 0.6


def test_post_referee_alert_rejects_unknown_type_and_bad_confidence(alert_client):
    unknown_type = alert_client.post("/api/referee-alerts", json={"type": "goal_scored"})
    out_of_range = alert_client.post(
        "/api/referee-alerts", json={"type": "foul_candidate", "confidence": 1.5}
    )

    assert unknown_type.status_code == 422
    assert out_of_range.status_code == 422


def test_manual_alert_is_visible_in_the_events_rest_list(alert_client):
    created = alert_client.post("/api/referee-alerts", json={"type": "offside_candidate"}).json()

    payload = alert_client.get("/api/events").json()
    matches = [event for event in payload["events"] if event["id"] == created["event_id"]]

    assert len(matches) == 1
    assert matches[0]["event_type"] == "offside_candidate"
    assert matches[0]["severity"] == "candidate"


def _make_alerts_app(
    auth_token: str | None = FIELD_TEST_TOKEN, root: Path | None = None
) -> FastAPI:
    application = FastAPI()
    application.state.referee_alerts = RefereeAlertService(StateStore())
    # The downlink shares the field-ingest bearer token boundary; attach a real
    # FieldIngestService so the websocket gate exercises the same authorize().
    application.state.field_ingest = FieldIngestService(
        FieldIngestSettings(
            auth_token=auth_token,
            root=root or Path(tempfile.mkdtemp(prefix="referee-alerts-test-")),
            srt_bind_host=None,
            srt_advertise_host=None,
            srt_port=10000,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    application.include_router(referee_alert_router)
    return application


def test_alert_socket_receives_alert_frame_and_records_acknowledgement():
    application = _make_alerts_app()
    service: RefereeAlertService = application.state.referee_alerts
    client = TestClient(application)
    alert = _alert(type=RefereeAlertType.OFFSIDE_CANDIDATE)

    with client.websocket_connect(
        "/ws/v1/field/alerts", headers={"Authorization": f"Bearer {FIELD_TEST_TOKEN}"}
    ) as websocket:
        # Publish on the session's own loop so the push is deterministic.
        assert websocket.portal.call(service.publish_referee_alert, alert) is True
        assert websocket.receive_json() == alert.model_dump(mode="json")

        websocket.send_json({"type": "acknowledged", "event_id": alert.event_id})
        # FIFO: receiving the error for this bogus frame proves the ack above
        # was already processed and recorded.
        websocket.send_json({"type": "nonsense"})
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert [ack.event_id for ack in service.acknowledgements] == [alert.event_id]

        # A duplicate event_id is a no-op: the next frame is the error reply,
        # never a second alert push.
        assert websocket.portal.call(service.publish_referee_alert, alert) is False
        websocket.send_json({"type": "nonsense"})
        assert websocket.receive_json()["type"] == "error"

        # Malformed JSON answers with an error frame and keeps the session.
        websocket.send_text("not-json")
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "INVALID_MESSAGE"

    deadline = time.time() + 2.0
    while time.time() < deadline and service.client_count:
        time.sleep(0.01)
    assert service.client_count == 0


def test_alert_socket_rejects_missing_or_wrong_bearer_token(tmp_path):
    application = _make_alerts_app(root=tmp_path)
    service: RefereeAlertService = application.state.referee_alerts
    client = TestClient(application)

    for headers in ({}, {"Authorization": "Bearer wrong-token"}):
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/ws/v1/field/alerts", headers=headers):
                pass  # pragma: no cover - the socket is closed before accept
        assert excinfo.value.code == 1008

    assert service.client_count == 0


def test_alert_socket_rejects_when_field_token_is_unconfigured(tmp_path):
    application = _make_alerts_app(auth_token=None, root=tmp_path)
    client = TestClient(application)

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(
            "/ws/v1/field/alerts", headers={"Authorization": f"Bearer {FIELD_TEST_TOKEN}"}
        ):
            pass  # pragma: no cover - the socket is closed before accept
    assert excinfo.value.code == 1008
