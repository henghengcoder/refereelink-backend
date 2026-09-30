from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from jsonschema import validate

from app.server.main import app
from app.state.models import (
    CaptureSourceMetadata,
    FrameState,
    GameEvent,
    HomographyStatus,
    PlayerRole,
    PlayerState,
)


SCHEMA_PATH = Path(__file__).parent / "contracts" / "frame_state.schema.json"


def _sample_frame() -> FrameState:
    return FrameState(
        frame_id=42,
        capture_timestamp_ms=1000.0,
        processed_timestamp_ms=1016.5,
        processing_fps=60.0,
        homography_status=HomographyStatus.UNAVAILABLE,
        players=[
            PlayerState(
                track_id=7,
                role=PlayerRole.UNKNOWN,
                team_id=-1,
                confidence=0.0,
            )
        ],
        events=[
            GameEvent(
                event_type="foul_candidate",
                confidence=0.4,
                severity="possible",
                timestamp=1.0165,
                frame_id=42,
            )
        ],
    )


def _validate_frame_payload(payload: dict) -> None:
    schema = json.loads(SCHEMA_PATH.read_text())
    validate(instance=payload, schema=schema)
    assert payload["type"] == "frame_state"
    assert isinstance(payload["frame_id"], int)
    assert payload["processed_timestamp_ms"] >= payload["capture_timestamp_ms"]
    for event in payload["events"]:
        assert event["event_type"]
        assert isinstance(event["timestamp"], (int, float))
        assert isinstance(event["frame_id"], int)


def test_frame_state_json_contract_preserves_unknown_and_unavailable() -> None:
    payload = json.loads(_sample_frame().model_dump_json())

    _validate_frame_payload(payload)
    assert payload["players"][0]["team"] == "unknown"
    assert payload["players"][0]["team_id"] == -1
    assert "ball" not in payload
    assert "possession_track_id" not in payload

    roundtrip = FrameState.model_validate_json(json.dumps(payload))
    assert roundtrip.frame_id == 42
    assert roundtrip.players[0].role is PlayerRole.UNKNOWN


def test_frame_state_contract_preserves_field_capture_metadata() -> None:
    frame = _sample_frame().model_copy(
        update={
            "capture_source": CaptureSourceMetadata(
                kind="field",
                session_id="session",
                stream_epoch=3,
                source_frame_id=42,
                t_us=1_200_000,
                transport_pts90k=108_000,
                camera_motion={"pitch": 0.2},
                pose_missing_reason=None,
                backend_received_at_ms=1005.0,
            )
        }
    )

    payload = json.loads(frame.model_dump_json())
    _validate_frame_payload(payload)
    assert payload["capture_source"]["stream_epoch"] == 3
    assert payload["capture_source"]["transport_pts90k"] == 108_000


def test_metrics_snapshot_json_contract_has_stable_disconnected_defaults() -> None:
    payload = json.loads(app.state.store.metrics.model_dump_json())

    assert payload["type"] == "metrics"
    assert payload["source_status"] == "disconnected"
    assert isinstance(payload["processing_fps"], (int, float))
    assert isinstance(payload["dropped_frames"], int)
    assert isinstance(payload["track_lifecycle_counts"], dict)


def test_websocket_frame_payload_obeys_contract() -> None:
    store = app.state.store
    previous_frame_state = store.latest_frame_state
    store.latest_frame_state = _sample_frame()
    try:
        with TestClient(app).websocket_connect("/ws/state") as websocket:
            for _ in range(5):
                payload = json.loads(websocket.receive_text())
                if payload.get("type") == "frame_state":
                    _validate_frame_payload(payload)
                    assert payload["frame_id"] == 42
                    break
            else:
                raise AssertionError("WebSocket did not publish a frame_state payload")
    finally:
        store.latest_frame_state = previous_frame_state
