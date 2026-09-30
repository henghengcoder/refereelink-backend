from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import re
from pydantic import BaseModel

from app.state.models import SourceStatus
from app.state.store import StateStore

router = APIRouter()
logger = logging.getLogger(__name__)


def get_store(request: Request) -> StateStore:
    return request.app.state.store


class PipelineStartPayload(BaseModel):
    """Payload for POST /api/pipeline/start.

    Any of the fields can be omitted. When ``video_source`` is provided
    and no pipeline is currently attached, the server builds a new
    InferencePipeline bound to that source and starts it. When omitted,
    the call falls back to the existing pipeline (if any).
    """

    video_source: Optional[str] = None
    field_session_id: Optional[str] = None
    field_stream_epoch: Optional[int] = None
    device: Optional[str] = None
    inference_backend: Optional[str] = None
    enable_foul_detection: Optional[bool] = None
    foul_checkpoint_path: Optional[str] = None
    foul_confidence_threshold: Optional[float] = None
    player_model_path: Optional[str] = None
    pitch_model_path: Optional[str] = None
    camera_calibration_path: Optional[str] = None
    enable_undistortion: Optional[bool] = None
    calibration_alpha: Optional[float] = None
    pitch_detection_interval: Optional[int] = None
    imgsz: Optional[int] = None
    player_confidence: Optional[float] = None
    player_iou: Optional[float] = None
    max_prediction_gap_frames: Optional[int] = None
    track_reactivation_window_frames: Optional[int] = None
    role_model_path: Optional[str] = None
    team_classifier_path: Optional[str] = None
    team_calibration_path: Optional[str] = None
    role_detection_interval: Optional[int] = None
    team_classification_interval: Optional[int] = None
    track_activation_threshold: Optional[float] = None
    track_lost_buffer: Optional[int] = None
    track_matching_threshold: Optional[float] = None
    track_minimum_consecutive_frames: Optional[int] = None
    enable_recording: Optional[bool] = None
    target_video_path: Optional[str] = None


@router.post("/api/pipeline/start")
async def pipeline_start(request: Request, payload: PipelineStartPayload) -> dict:
    store = get_store(request)
    calibration = store.team_calibration
    if store.config.require_team_calibration and not calibration.can_run:
        return JSONResponse(
            status_code=409,
            content={
                "status": "error",
                "code": "TEAM_CALIBRATION_REQUIRED",
                "detail": "complete and validate pre-match team calibration before starting",
                "calibration_state": calibration.state.value,
            },
        )
    if calibration.state.value == "ready":
        calibration.mark_running()
    create_pipeline = getattr(request.app.state, "create_pipeline", None)
    attach_and_start = getattr(request.app.state, "attach_and_start_pipeline", None)
    current = getattr(request.app.state, "pipeline", None)

    device = payload.device or store.config.device or "cpu"
    config_requested = any(
        value is not None
        for value in (
            payload.player_model_path,
            payload.pitch_model_path,
            payload.camera_calibration_path,
            payload.enable_undistortion,
            payload.calibration_alpha,
            payload.pitch_detection_interval,
            payload.imgsz,
            payload.player_confidence,
            payload.player_iou,
            payload.max_prediction_gap_frames,
            payload.track_reactivation_window_frames,
            payload.role_model_path,
            payload.team_classifier_path,
            payload.team_calibration_path,
            payload.role_detection_interval,
            payload.team_classification_interval,
            payload.track_activation_threshold,
            payload.track_lost_buffer,
            payload.track_matching_threshold,
            payload.track_minimum_consecutive_frames,
            payload.enable_recording,
            payload.target_video_path,
            payload.inference_backend,
            payload.enable_foul_detection,
            payload.foul_checkpoint_path,
            payload.foul_confidence_threshold,
            payload.field_session_id,
            payload.field_stream_epoch,
        )
    )

    if current is not None and (
        payload.video_source is None
        and payload.field_session_id is None
        and payload.field_stream_epoch is None
        and payload.device is None
        and payload.inference_backend is None
        and payload.enable_foul_detection is None
        and payload.foul_checkpoint_path is None
        and payload.foul_confidence_threshold is None
        and payload.player_model_path is None
        and payload.pitch_model_path is None
        and payload.camera_calibration_path is None
        and payload.enable_undistortion is None
        and payload.calibration_alpha is None
        and payload.pitch_detection_interval is None
        and payload.imgsz is None
        and payload.player_confidence is None
        and payload.player_iou is None
        and payload.max_prediction_gap_frames is None
        and payload.track_reactivation_window_frames is None
        and payload.role_model_path is None
        and payload.team_classifier_path is None
        and payload.team_calibration_path is None
        and payload.role_detection_interval is None
        and payload.team_classification_interval is None
        and payload.track_activation_threshold is None
        and payload.track_lost_buffer is None
        and payload.track_matching_threshold is None
        and payload.track_minimum_consecutive_frames is None
        and payload.enable_recording is None
        and payload.target_video_path is None
    ):
        # Resume existing pipeline.
        store.pipeline_running = True
        try:
            current.start()
        except Exception:
            pass
        return {
            "status": "started",
            "mode": "resume",
            "recording": current.recording_status,
        }

    field_requested = payload.field_session_id is not None or payload.field_stream_epoch is not None
    if field_requested and (
        payload.field_session_id is None
        or payload.field_stream_epoch is None
        or payload.video_source is not None
    ):
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "detail": "choose either video_source or field session/epoch",
            },
        )
    requested_source = (
        None if field_requested else (payload.video_source or store.config.video_source)
    )
    should_create = (field_requested and (current is None or config_requested)) or (
        not field_requested
        and bool(requested_source)
        and (
            current is None or payload.video_source != store.config.video_source or config_requested
        )
    )
    if should_create:
        # Build a new pipeline with the requested source.
        try:
            if create_pipeline is None or attach_and_start is None:
                return {"status": "error", "detail": "pipeline factory not registered"}
            pipeline = create_pipeline(
                requested_source,
                field_session_id=payload.field_session_id,
                field_stream_epoch=payload.field_stream_epoch,
                device=device,
                inference_backend=payload.inference_backend or store.config.inference_backend,
                enable_foul_detection=(
                    payload.enable_foul_detection
                    if payload.enable_foul_detection is not None
                    else store.config.enable_foul_detection
                ),
                foul_checkpoint_path=payload.foul_checkpoint_path,
                foul_confidence_threshold=(
                    payload.foul_confidence_threshold
                    if payload.foul_confidence_threshold is not None
                    else store.config.foul_confidence_threshold
                ),
                player_model_path=payload.player_model_path or store.config.player_model_path,
                pitch_model_path=payload.pitch_model_path or store.config.pitch_model_path,
                camera_calibration_path=(
                    payload.camera_calibration_path
                    if payload.camera_calibration_path is not None
                    else store.config.camera_calibration_path
                ),
                enable_undistortion=(
                    payload.enable_undistortion
                    if payload.enable_undistortion is not None
                    else store.config.enable_undistortion
                ),
                calibration_alpha=(
                    payload.calibration_alpha
                    if payload.calibration_alpha is not None
                    else store.config.calibration_alpha
                ),
                pitch_detection_interval=(
                    payload.pitch_detection_interval
                    if payload.pitch_detection_interval is not None
                    else store.config.pitch_detection_interval
                ),
                imgsz=payload.imgsz if payload.imgsz is not None else store.config.imgsz,
                player_confidence=(
                    payload.player_confidence
                    if payload.player_confidence is not None
                    else store.config.player_confidence
                ),
                player_iou=(
                    payload.player_iou
                    if payload.player_iou is not None
                    else store.config.player_iou
                ),
                max_prediction_gap_frames=(
                    payload.max_prediction_gap_frames
                    if payload.max_prediction_gap_frames is not None
                    else store.config.max_prediction_gap_frames
                ),
                track_reactivation_window_frames=(
                    payload.track_reactivation_window_frames
                    if payload.track_reactivation_window_frames is not None
                    else store.config.track_reactivation_window_frames
                ),
                role_model_path=payload.role_model_path or store.config.role_model_path,
                team_classifier_path=(
                    payload.team_classifier_path
                    if payload.team_classifier_path is not None
                    else store.config.team_classifier_path
                ),
                team_calibration_path=(
                    payload.team_calibration_path
                    if payload.team_calibration_path is not None
                    else store.config.team_calibration_path
                ),
                role_detection_interval=(
                    payload.role_detection_interval
                    if payload.role_detection_interval is not None
                    else store.config.role_detection_interval
                ),
                team_classification_interval=(
                    payload.team_classification_interval
                    if payload.team_classification_interval is not None
                    else store.config.team_classification_interval
                ),
                track_activation_threshold=(
                    payload.track_activation_threshold
                    if payload.track_activation_threshold is not None
                    else store.config.track_activation_threshold
                ),
                track_lost_buffer=(
                    payload.track_lost_buffer
                    if payload.track_lost_buffer is not None
                    else store.config.track_lost_buffer
                ),
                track_matching_threshold=(
                    payload.track_matching_threshold
                    if payload.track_matching_threshold is not None
                    else store.config.track_matching_threshold
                ),
                track_minimum_consecutive_frames=(
                    payload.track_minimum_consecutive_frames
                    if payload.track_minimum_consecutive_frames is not None
                    else store.config.track_minimum_consecutive_frames
                ),
                enable_recording=(
                    payload.enable_recording
                    if payload.enable_recording is not None
                    else store.config.enable_recording
                ),
                target_video_path=(
                    payload.target_video_path
                    if payload.target_video_path is not None
                    else store.config.target_video_path
                ),
            )
            attach_and_start(pipeline)
            store.update_config(
                {
                    "video_source": requested_source or "",
                    "device": device,
                    "inference_backend": payload.inference_backend
                    or store.config.inference_backend,
                    "foul_confidence_threshold": (
                        payload.foul_confidence_threshold
                        if payload.foul_confidence_threshold is not None
                        else store.config.foul_confidence_threshold
                    ),
                    "player_model_path": payload.player_model_path
                    or store.config.player_model_path,
                    "pitch_model_path": payload.pitch_model_path or store.config.pitch_model_path,
                    "camera_calibration_path": (
                        payload.camera_calibration_path
                        if payload.camera_calibration_path is not None
                        else store.config.camera_calibration_path
                    ),
                    "enable_undistortion": (
                        payload.enable_undistortion
                        if payload.enable_undistortion is not None
                        else store.config.enable_undistortion
                    ),
                    "calibration_alpha": (
                        payload.calibration_alpha
                        if payload.calibration_alpha is not None
                        else store.config.calibration_alpha
                    ),
                    "pitch_detection_interval": (
                        payload.pitch_detection_interval
                        if payload.pitch_detection_interval is not None
                        else store.config.pitch_detection_interval
                    ),
                    "imgsz": payload.imgsz if payload.imgsz is not None else store.config.imgsz,
                    "player_confidence": (
                        payload.player_confidence
                        if payload.player_confidence is not None
                        else store.config.player_confidence
                    ),
                    "player_iou": (
                        payload.player_iou
                        if payload.player_iou is not None
                        else store.config.player_iou
                    ),
                    "max_prediction_gap_frames": (
                        payload.max_prediction_gap_frames
                        if payload.max_prediction_gap_frames is not None
                        else store.config.max_prediction_gap_frames
                    ),
                    "track_reactivation_window_frames": (
                        payload.track_reactivation_window_frames
                        if payload.track_reactivation_window_frames is not None
                        else store.config.track_reactivation_window_frames
                    ),
                    "role_model_path": payload.role_model_path or store.config.role_model_path,
                    "team_classifier_path": (
                        payload.team_classifier_path
                        if payload.team_classifier_path is not None
                        else store.config.team_classifier_path
                    ),
                    "team_calibration_path": (
                        payload.team_calibration_path
                        if payload.team_calibration_path is not None
                        else store.config.team_calibration_path
                    ),
                    "role_detection_interval": (
                        payload.role_detection_interval
                        if payload.role_detection_interval is not None
                        else store.config.role_detection_interval
                    ),
                    "team_classification_interval": (
                        payload.team_classification_interval
                        if payload.team_classification_interval is not None
                        else store.config.team_classification_interval
                    ),
                    "track_activation_threshold": (
                        payload.track_activation_threshold
                        if payload.track_activation_threshold is not None
                        else store.config.track_activation_threshold
                    ),
                    "track_lost_buffer": (
                        payload.track_lost_buffer
                        if payload.track_lost_buffer is not None
                        else store.config.track_lost_buffer
                    ),
                    "track_matching_threshold": (
                        payload.track_matching_threshold
                        if payload.track_matching_threshold is not None
                        else store.config.track_matching_threshold
                    ),
                    "track_minimum_consecutive_frames": (
                        payload.track_minimum_consecutive_frames
                        if payload.track_minimum_consecutive_frames is not None
                        else store.config.track_minimum_consecutive_frames
                    ),
                    "enable_recording": (
                        payload.enable_recording
                        if payload.enable_recording is not None
                        else store.config.enable_recording
                    ),
                    "target_video_path": (
                        payload.target_video_path
                        if payload.target_video_path is not None
                        else store.config.target_video_path
                    ),
                    "enable_foul_detection": bool(payload.enable_foul_detection)
                    if payload.enable_foul_detection is not None
                    else store.config.enable_foul_detection,
                }
            )
            return {
                "status": "started",
                "mode": "new",
                "video_source": requested_source,
                "field_session_id": payload.field_session_id,
                "field_stream_epoch": payload.field_stream_epoch,
                "recording": pipeline.recording_status,
            }
        except FileNotFoundError as exc:
            store.source_status = SourceStatus.ERROR
            return {"status": "error", "detail": f"video source not found: {exc}"}
        except (KeyError, PermissionError) as exc:
            # A field epoch is an explicitly allocated, single-consumer live
            # resource.  Make stale, released, or already-bound epochs a
            # client-visible conflict instead of returning a 200 error body.
            # The detail is a fixed string so the exception text stays in logs.
            logger.info("Rejected pipeline start: %s", exc)
            detail = (
                "live epoch not found"
                if isinstance(exc, KeyError)
                else "live epoch already has an inference consumer"
            )
            store.source_status = SourceStatus.ERROR
            if field_requested:
                return JSONResponse(
                    status_code=409,
                    content={"status": "error", "detail": detail},
                )
            return {"status": "error", "detail": detail}
        except Exception as exc:
            logger.exception("Failed to start pipeline with source=%s", requested_source)
            store.source_status = SourceStatus.ERROR
            return {"status": "error", "detail": str(exc)}

    if current is not None:
        store.pipeline_running = True
        try:
            current.start()
        except Exception:
            pass
        return {"status": "started", "mode": "resume"}

    return {
        "status": "error",
        "detail": "no video_source provided and no existing pipeline to resume",
    }


@router.post("/api/pipeline/stop")
async def pipeline_stop(request: Request) -> dict:
    store = get_store(request)
    stop_and_clear = getattr(request.app.state, "stop_and_clear_pipeline", None)
    if stop_and_clear is not None:
        stop_and_clear()
        return {"status": "stopped", "mode": "cleared"}
    store.pipeline_running = False
    store.source_status = SourceStatus.DISCONNECTED
    return {"status": "stopped"}


@router.get("/api/pipeline/recording")
async def pipeline_recording(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    path_value = getattr(pipeline, "recording_path", None)
    path = Path(path_value) if path_value else None
    if path is None or not path.is_file():
        return JSONResponse(status_code=404, content={"detail": "recording is not available"})
    return _video_response(path, request)


def _video_response(path: Path, request: Request):
    """Serve a debug recording with HTTP Range support for browser seeking."""

    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "video/mp4",
    }
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers=headers)

    match = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if match is None:
        return JSONResponse(status_code=416, content={"detail": "invalid range"})
    start_text, end_text = match.groups()
    if start_text == "" and end_text == "":
        return JSONResponse(status_code=416, content={"detail": "invalid range"})
    if start_text == "":
        length = min(int(end_text), size)
        start, end = max(size - length, 0), size - 1
    else:
        start = int(start_text)
        end = min(int(end_text), size - 1) if end_text else size - 1
    if start >= size or start > end:
        return JSONResponse(status_code=416, content={"detail": "range not satisfiable"})

    def iterator():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers.update(
        {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        }
    )
    return StreamingResponse(iterator(), status_code=206, headers=headers, media_type="video/mp4")
