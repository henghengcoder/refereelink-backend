from __future__ import annotations

import argparse
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.pipeline.buffer import PipelineMode
from app.pipeline.engine import InferencePipeline
from app.pipeline.source import create_video_source
from app.referee_alerts.api import router as referee_alert_router
from app.referee_alerts.service import RefereeAlertService
from app.field_ingest.api import create_router as create_field_ingest_router
from app.field_ingest.service import FieldIngestService
from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
    ROLE_DETECTION_MODEL_PATH,
    TEAM_CLASSIFIER_PATH,
)
from app.server.api.health import router as health_router
from app.server.api.status import router as status_router
from app.server.api.events import router as events_router
from app.server.api.multiview import router as multiview_router
from app.server.api.pipeline import router as pipeline_router
from app.server.api.team_calibration import router as team_calibration_router
from app.server.ws.state import router as ws_router
from app.services.publisher import WebSocketPublisher
from app.multiview.live_ingest.service import LiveMultiviewService
from app.multiview.repository import MultiviewCaseRepository
from app.multiview.service import MultiviewAnalysisService
from app.state.models import SourceStatus
from app.state.store import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_store = StateStore()
_publisher = WebSocketPublisher(_store)
_referee_alerts = RefereeAlertService(_store, _publisher)
_pipeline: Optional[InferencePipeline] = None
_pipeline_lock = threading.Lock()
_device = "cpu"
_foul_checkpoint_path: Optional[str] = None


def _put_placeholder(text: str) -> None:
    """Write a black placeholder frame with text into the store so the
    MJPEG endpoint always has something to serve."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        text,
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Configure a video source and press START",
        (40, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    _store.publish_raw_frame(frame)


def _generate_mjpeg() -> Iterator[bytes]:
    while True:
        jpeg = _store.latest_jpeg_frame
        if jpeg is None:
            _put_placeholder("NO VIDEO SOURCE")
            jpeg = _store.latest_jpeg_frame
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(1.0 / 30.0)


def create_pipeline(
    video_source: Optional[str] = None,
    field_session_id: Optional[str] = None,
    field_stream_epoch: Optional[int] = None,
    device: str = "cpu",
    inference_backend: str = "auto",
    enable_foul_detection: bool = False,
    foul_checkpoint_path: Optional[str] = None,
    foul_confidence_threshold: float = 0.48,
    player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
    pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
    camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
    enable_undistortion: bool = True,
    calibration_alpha: float = 0.0,
    pitch_detection_interval: int = 5,
    imgsz: int = 640,
    player_confidence: float = 0.25,
    player_iou: float = 0.7,
    max_prediction_gap_frames: int = 6,
    track_reactivation_window_frames: int = 12,
    role_model_path: str = ROLE_DETECTION_MODEL_PATH,
    team_classifier_path: Optional[str] = TEAM_CLASSIFIER_PATH,
    team_calibration_path: Optional[str] = None,
    role_detection_interval: int = 3,
    team_classification_interval: int = 5,
    track_activation_threshold: float = 0.25,
    track_lost_buffer: int = 45,
    track_matching_threshold: float = 0.8,
    track_minimum_consecutive_frames: int = 2,
    enable_recording: bool = False,
    target_video_path: Optional[str] = None,
    calibration_session: Optional[object] = None,
) -> InferencePipeline:
    """Factory used by both CLI startup and the REST API to build a
    pipeline bound to the shared store."""
    if field_session_id is not None or field_stream_epoch is not None:
        if field_session_id is None or field_stream_epoch is None or video_source is not None:
            raise ValueError("choose either video_source or field session/epoch")
        source = app.state.field_ingest.open_frame_source(
            field_session_id,
            field_stream_epoch,
            store=_store,
        )
    else:
        if not video_source:
            raise ValueError("video_source or field session/epoch is required")
        source = create_video_source(video_source, store=_store)
    try:
        return InferencePipeline(
            source=source,
            store=_store,
            device=device,
            inference_backend=inference_backend,
            enable_foul_detection=enable_foul_detection,
            foul_checkpoint_path=foul_checkpoint_path,
            foul_confidence_threshold=foul_confidence_threshold,
            mode=PipelineMode.REALTIME,
            player_model_path=player_model_path,
            pitch_model_path=pitch_model_path,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            calibration_alpha=calibration_alpha,
            pitch_detection_interval=pitch_detection_interval,
            imgsz=imgsz,
            player_confidence=player_confidence,
            player_iou=player_iou,
            max_prediction_gap_frames=max_prediction_gap_frames,
            track_reactivation_window_frames=track_reactivation_window_frames,
            role_model_path=role_model_path,
            team_classifier_path=team_classifier_path,
            team_calibration_path=team_calibration_path,
            role_detection_interval=role_detection_interval,
            team_classification_interval=team_classification_interval,
            track_activation_threshold=track_activation_threshold,
            track_lost_buffer=track_lost_buffer,
            track_matching_threshold=track_matching_threshold,
            track_minimum_consecutive_frames=track_minimum_consecutive_frames,
            enable_recording=enable_recording,
            target_video_path=target_video_path,
            frame_observer=(
                getattr(calibration_session, "observe_frame", None)
                if calibration_session is not None
                else None
            ),
        )
    except Exception:
        # open_frame_source already claimed the only inference consumer. A
        # constructor failure leaves no pipeline for /api/pipeline/stop to
        # release, which would block every later live allocation.
        try:
            source.release()
        except Exception:
            logger.exception("Failed to release source after pipeline construction error")
        raise


def attach_and_start_pipeline(pipeline: InferencePipeline) -> None:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception:
                pass
        _pipeline = pipeline
        app.state.pipeline = _pipeline
    _store.pipeline_running = True
    try:
        pipeline.start()
    except Exception:
        _store.pipeline_running = False
        try:
            pipeline.stop()
        except Exception:
            logger.exception("Failed to release pipeline after start error")
        with _pipeline_lock:
            if _pipeline is pipeline:
                _pipeline = None
                app.state.pipeline = None
        raise


def stop_and_clear_pipeline() -> None:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception:
                pass
            _pipeline = None
        app.state.pipeline = None
    _store._latest_raw_frame = None  # type: ignore[attr-defined]
    _store._jpeg_frame.clear()
    _store.source_status = SourceStatus.DISCONNECTED
    _store.pipeline_running = False
    _put_placeholder("PIPELINE STOPPED")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    _put_placeholder("SERVER READY")
    yield
    logger.info("Server shutting down...")
    stop_and_clear_pipeline()


app = FastAPI(title="RefereeLink Backend Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.store = _store
app.state.publisher = _publisher
app.state.pipeline = None
app.state.live_multiview_service = LiveMultiviewService.from_environment()
app.state.multiview_service = MultiviewAnalysisService(
    repository=MultiviewCaseRepository(live_case_store=app.state.live_multiview_service.case_store)
)
app.state.create_pipeline = create_pipeline
app.state.attach_and_start_pipeline = attach_and_start_pipeline
app.state.stop_and_clear_pipeline = stop_and_clear_pipeline
app.state.field_ingest = FieldIngestService()
app.state.referee_alerts = _referee_alerts

app.include_router(health_router)
app.include_router(status_router)
app.include_router(events_router)
app.include_router(multiview_router)
app.include_router(pipeline_router)
app.include_router(team_calibration_router)
app.include_router(referee_alert_router)
app.include_router(ws_router)
app.include_router(create_field_ingest_router(app.state.field_ingest))


@app.get("/video/stream")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def main() -> None:
    global _device

    parser = argparse.ArgumentParser(description="RefereeLink Backend Server")
    parser.add_argument(
        "--video_source",
        type=str,
        default=None,
        help="Video file path or RTSP URL. If omitted, the "
        "server starts idle and waits for a source "
        "from the web UI.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference (cpu, cuda)"
    )
    parser.add_argument(
        "--inference_backend",
        type=str,
        choices=("auto", "pytorch", "onnx", "tensorrt"),
        default="auto",
    )
    parser.add_argument("--player_model_path", type=str, default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch_model_path", type=str, default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--camera_calibration_path", type=str, default=CAMERA_CALIBRATION_PATH)
    parser.add_argument("--disable_undistortion", action="store_false", dest="enable_undistortion")
    parser.set_defaults(enable_undistortion=True)
    parser.add_argument("--calibration_alpha", type=float, default=0.0)
    parser.add_argument("--pitch_detection_interval", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--player_confidence", type=float, default=0.25)
    parser.add_argument("--player_iou", type=float, default=0.7)
    parser.add_argument("--max_prediction_gap_frames", type=int, default=6)
    parser.add_argument("--track_reactivation_window_frames", type=int, default=12)
    parser.add_argument("--role_model_path", type=str, default=ROLE_DETECTION_MODEL_PATH)
    parser.add_argument("--team_classifier_path", type=str, default=None)
    parser.add_argument("--team_calibration_path", type=str, default=None)
    parser.add_argument("--role_detection_interval", type=int, default=3)
    parser.add_argument("--team_classification_interval", type=int, default=5)
    parser.add_argument("--track_activation_threshold", type=float, default=0.25)
    parser.add_argument("--track_lost_buffer", type=int, default=45)
    parser.add_argument("--track_matching_threshold", type=float, default=0.8)
    parser.add_argument("--track_minimum_consecutive_frames", type=int, default=2)
    parser.add_argument("--enable_recording", action="store_true")
    parser.add_argument("--target_video_path", type=str, default="")
    parser.add_argument("--enable_foul_detection", action="store_true")
    parser.add_argument("--foul_checkpoint_path", type=str, default=None)
    parser.add_argument("--foul_confidence_threshold", type=float, default=0.48)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _device = args.device
    _store.update_config(
        {
            "video_source": args.video_source or "",
            "device": args.device,
            "inference_backend": args.inference_backend,
            "enable_foul_detection": args.enable_foul_detection,
            "foul_confidence_threshold": args.foul_confidence_threshold,
            "player_model_path": args.player_model_path,
            "pitch_model_path": args.pitch_model_path,
            "camera_calibration_path": args.camera_calibration_path,
            "enable_undistortion": args.enable_undistortion,
            "calibration_alpha": args.calibration_alpha,
            "pitch_detection_interval": args.pitch_detection_interval,
            "imgsz": args.imgsz,
            "player_confidence": args.player_confidence,
            "player_iou": args.player_iou,
            "max_prediction_gap_frames": args.max_prediction_gap_frames,
            "track_reactivation_window_frames": args.track_reactivation_window_frames,
            "role_model_path": args.role_model_path,
            "team_classifier_path": args.team_classifier_path,
            "team_calibration_path": args.team_calibration_path,
            "role_detection_interval": args.role_detection_interval,
            "team_classification_interval": args.team_classification_interval,
            "track_activation_threshold": args.track_activation_threshold,
            "track_lost_buffer": args.track_lost_buffer,
            "track_matching_threshold": args.track_matching_threshold,
            "track_minimum_consecutive_frames": args.track_minimum_consecutive_frames,
            "enable_recording": args.enable_recording,
            "target_video_path": args.target_video_path,
        }
    )

    if args.video_source:
        pipeline = create_pipeline(
            args.video_source,
            device=_device,
            inference_backend=args.inference_backend,
            enable_foul_detection=args.enable_foul_detection,
            foul_checkpoint_path=args.foul_checkpoint_path,
            foul_confidence_threshold=args.foul_confidence_threshold,
            player_model_path=args.player_model_path,
            pitch_model_path=args.pitch_model_path,
            camera_calibration_path=args.camera_calibration_path,
            enable_undistortion=args.enable_undistortion,
            calibration_alpha=args.calibration_alpha,
            pitch_detection_interval=args.pitch_detection_interval,
            imgsz=args.imgsz,
            player_confidence=args.player_confidence,
            player_iou=args.player_iou,
            max_prediction_gap_frames=args.max_prediction_gap_frames,
            track_reactivation_window_frames=args.track_reactivation_window_frames,
            role_model_path=args.role_model_path,
            team_classifier_path=args.team_classifier_path,
            team_calibration_path=args.team_calibration_path,
            role_detection_interval=args.role_detection_interval,
            team_classification_interval=args.team_classification_interval,
            track_activation_threshold=args.track_activation_threshold,
            track_lost_buffer=args.track_lost_buffer,
            track_matching_threshold=args.track_matching_threshold,
            track_minimum_consecutive_frames=args.track_minimum_consecutive_frames,
            enable_recording=args.enable_recording,
            target_video_path=args.target_video_path or None,
        )
        attach_and_start_pipeline(pipeline)
    else:
        logger.info(
            "No --video_source provided; server starting idle. "
            "Configure a source from the web dashboard."
        )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
