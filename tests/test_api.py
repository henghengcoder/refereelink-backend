from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.server.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "pipeline_running" in data
    assert "source_status" in data
    assert "metrics" in data
    assert data["recording"]["enabled"] is False


def test_get_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "mode" in data
    assert "video_source" in data
    assert "device" in data
    assert "player_model_path" in data
    assert "pitch_model_path" in data
    assert "enable_ball" not in data
    assert "ball_model_path" not in data
    assert "ball_detection_interval" not in data
    assert data["inference_backend"] == "auto"


def test_update_config(client):
    r = client.put("/api/config", json={"device": "cuda", "mode": "realtime"})
    assert r.status_code == 200
    data = r.json()
    assert data["device"] == "cuda"
    assert data["mode"] == "realtime"


def test_get_events(client):
    r = client.get("/api/events")
    assert r.status_code == 200
    data = r.json()
    assert "events" in data
    assert "total" in data


def test_missing_field_epoch_returns_stable_conflict(client):
    store = client.app.state.store
    previous = store.config.require_team_calibration
    store.config.require_team_calibration = False
    try:
        response = client.post(
            "/api/pipeline/start",
            json={"field_session_id": str(uuid4()), "field_stream_epoch": 1},
        )
    finally:
        store.config.require_team_calibration = previous
    assert response.status_code == 409
    assert response.json()["detail"] == "live epoch not found"
    assert "KeyError" not in response.text
    assert "Traceback" not in response.text


def test_pipeline_start_stop(client):
    r = client.post("/api/pipeline/start", json={})
    assert r.status_code == 409
    status = r.json().get("status")
    assert status == "error"
    assert r.json().get("code") == "TEAM_CALIBRATION_REQUIRED"
    r = client.post("/api/pipeline/stop")
    assert r.status_code == 200
    assert r.json().get("status") == "stopped"


def test_video_stream_endpoint_exists(client):
    # Verify the /video/stream route is registered. We don't actually
    # call it here because the MJPEG generator is an infinite loop and
    # the test client would block.
    from app.server.main import app

    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/video/stream" in paths
    recording = client.get("/api/pipeline/recording")
    assert recording.status_code == 404


def test_recording_endpoint_supports_http_ranges(client, tmp_path):
    target = tmp_path / "annotated.mp4"
    target.write_bytes(b"0123456789")
    app.state.pipeline = type("RecordingPipeline", (), {"recording_path": str(target)})()
    try:
        response = client.get("/api/pipeline/recording", headers={"Range": "bytes=2-5"})
    finally:
        app.state.pipeline = None

    assert response.status_code == 206
    assert response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["accept-ranges"] == "bytes"
