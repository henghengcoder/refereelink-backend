from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from app.classification.team_calibration.session import TeamCalibrationSession
from app.classification.team_calibration.types import CalibrationLabel
from app.state.store import StateStore

router = APIRouter()
logger = logging.getLogger(__name__)


class CalibrationStartPayload(BaseModel):
    match_id: str
    camera_id: str = "default"
    video_source: Optional[str] = None
    bundle_path: Optional[str] = None
    device: Optional[str] = None


class CalibrationLabelPayload(BaseModel):
    track_id: int
    label: CalibrationLabel


class SourcePreviewPayload(BaseModel):
    video_source: Optional[str] = None
    match_id: str = "match-1"
    camera_id: str = "camera-1"
    bundle_path: Optional[str] = None
    device: Optional[str] = None


class ClipStartPayload(BaseModel):
    start_ms: int


class ClipFinishPayload(BaseModel):
    end_ms: int


def _session(request: Request) -> TeamCalibrationSession:
    store: StateStore = request.app.state.store
    return store.team_calibration


@router.get("/api/team-calibration")
async def get_team_calibration(request: Request) -> dict:
    return _session(request).snapshot()


@router.post("/api/team-calibration/source/preview")
async def preview_team_calibration_source(
    request: Request,
    payload: SourcePreviewPayload,
) -> dict:
    store: StateStore = request.app.state.store
    source = payload.video_source or store.config.video_source
    if not source:
        raise HTTPException(status_code=400, detail="video_source is required for calibration")
    device = payload.device or store.config.device
    try:
        snapshot = store.team_calibration_clip.preview(
            source_path=source,
            match_id=payload.match_id,
            camera_id=payload.camera_id,
            device=device,
            bundle_path=payload.bundle_path,
            config=store.config,
        )
        info = store.team_calibration_clip.source_info
        assert info is not None
        store.update_config({"video_source": source, "device": device})
        return {
            **snapshot,
            "source_url": "/api/team-calibration/clip/source",
            "duration_ms": info.duration_ms,
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/team-calibration/clip/start")
async def start_team_calibration_clip(
    request: Request,
    payload: ClipStartPayload,
) -> dict:
    try:
        return request.app.state.store.team_calibration_clip.start_clip(payload.start_ms)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/team-calibration/clip/finish")
async def finish_team_calibration_clip(
    request: Request,
    payload: ClipFinishPayload,
) -> dict:
    store: StateStore = request.app.state.store
    try:
        return store.team_calibration_clip.finish_clip(payload.end_ms)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/team-calibration/clip/status")
async def get_team_calibration_clip_status(request: Request) -> dict:
    return _session(request).snapshot()


@router.get("/api/team-calibration/clip/metadata")
async def get_team_calibration_clip_metadata(request: Request) -> dict:
    try:
        return request.app.state.store.team_calibration_clip.metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/team-calibration/clip/source")
async def get_team_calibration_source_video(request: Request):
    path = request.app.state.store.team_calibration_clip.source_path
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="source preview is not loaded")
    return _video_response(path, request)


@router.get("/api/team-calibration/clip/video")
async def get_team_calibration_clip_video(request: Request):
    try:
        path = request.app.state.store.team_calibration_clip.review_video_path()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _video_response(path, request)


@router.post("/api/team-calibration/start")
async def start_team_calibration(
    request: Request,
    payload: CalibrationStartPayload,
) -> dict:
    store: StateStore = request.app.state.store
    session = store.team_calibration
    source = payload.video_source or store.config.video_source
    if not source:
        raise HTTPException(status_code=400, detail="video_source is required for calibration")
    create_pipeline = getattr(request.app.state, "create_pipeline", None)
    attach_and_start = getattr(request.app.state, "attach_and_start_pipeline", None)
    if create_pipeline is None or attach_and_start is None:
        raise HTTPException(status_code=503, detail="pipeline factory not registered")

    session.start(
        match_id=payload.match_id,
        camera_id=payload.camera_id,
        bundle_path=payload.bundle_path,
        device=payload.device or store.config.device,
    )
    try:
        pipeline = create_pipeline(
            source,
            device=payload.device or store.config.device,
            inference_backend=store.config.inference_backend,
            enable_foul_detection=False,
            player_model_path=store.config.player_model_path,
            pitch_model_path=store.config.pitch_model_path,
            camera_calibration_path=store.config.camera_calibration_path,
            enable_undistortion=store.config.enable_undistortion,
            calibration_alpha=store.config.calibration_alpha,
            pitch_detection_interval=store.config.pitch_detection_interval,
            imgsz=store.config.imgsz,
            role_model_path=store.config.role_model_path,
            team_calibration_path=None,
            role_detection_interval=store.config.role_detection_interval,
            team_classification_interval=store.config.team_classification_interval,
            calibration_session=session,
        )
        attach_and_start(pipeline)
        store.update_config(
            {"video_source": source, "device": payload.device or store.config.device}
        )
    except Exception as exc:
        logger.exception("Failed to start team calibration")
        session.reset()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return session.snapshot()


@router.post("/api/team-calibration/label")
async def label_team_track(
    request: Request,
    payload: CalibrationLabelPayload,
) -> dict:
    try:
        return request.app.state.store.team_calibration_clip.label_track(
            payload.track_id,
            payload.label,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/team-calibration/validate")
async def validate_team_calibration(request: Request) -> dict:
    try:
        store: StateStore = request.app.state.store
        snapshot = store.team_calibration.validate()
        if snapshot.get("state") == "ready" and snapshot.get("bundle_path"):
            store.update_config({"team_calibration_path": snapshot["bundle_path"]})
        return snapshot
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/team-calibration/reset")
async def reset_team_calibration(request: Request) -> dict:
    try:
        return request.app.state.store.team_calibration_clip.reset()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _video_response(path, request: Request):
    """Serve a controlled local video with HTTP Range support for seeking."""

    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
    }
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers=headers)
    match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())
    if match is None:
        raise HTTPException(status_code=416, detail="invalid byte range")
    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) else size - 1
    if start >= size or end < start:
        raise HTTPException(status_code=416, detail="byte range is outside the video")
    end = min(end, size - 1)
    length = end - start + 1
    headers.update(
        {
            "Content-Length": str(length),
            "Content-Range": f"bytes {start}-{end}/{size}",
        }
    )

    def iterator():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(iterator(), status_code=206, headers=headers, media_type="video/mp4")
