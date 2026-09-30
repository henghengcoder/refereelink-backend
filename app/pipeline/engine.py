from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
    ROLE_DETECTION_MODEL_PATH,
    TEAM_CLASSIFIER_PATH,
    FOUL_MODEL_PATH,
)
from app.events.engine import EventEngine, FoulEventAdapter
from app.pipeline.buffer import BoundedFrameBuffer, PipelineMode
from app.field_ingest.frames import CapturedFrame
from app.pipeline.recorder import VideoRecorder
from app.pipeline.source import VideoSource
from app.state.models import (
    FrameState,
    HomographyStatus,
    MetricsSnapshot,
    PlayerRole,
    PlayerState,
    SourceStatus,
    TeamLabel,
)
from app.state.store import StateStore
from app.vision.core import VisionCore
from app.vision.display import TrackDisplaySmoother

logger = logging.getLogger(__name__)

METRICS_INTERVAL_SEC = 1.0

# Compact, high-contrast colours used by the final video overlay.  These are
# BGR values because the pipeline renders with OpenCV.
HOME_OVERLAY_COLOR = (147, 20, 255)  # pink / magenta
AWAY_OVERLAY_COLOR = (255, 191, 0)  # cyan / blue
REFEREE_OVERLAY_COLOR = (0, 215, 255)  # yellow
UNKNOWN_OVERLAY_COLOR = (170, 170, 170)


def _capture_source_metadata(captured: CapturedFrame | None):
    if captured is None or not captured.session_id:
        return None
    received_ms: float | None = None
    try:
        received_ms = (
            datetime.fromisoformat(captured.backend_received_at.replace("Z", "+00:00")).timestamp()
            * 1000.0
        )
    except (TypeError, ValueError, AttributeError):
        pass
    from app.state.models import CaptureSourceMetadata

    return CaptureSourceMetadata(
        kind="field",
        session_id=captured.session_id,
        stream_epoch=captured.stream_epoch,
        source_frame_id=captured.source_frame_id,
        t_us=captured.t_us,
        transport_pts90k=captured.transport_pts90k,
        camera_motion=captured.camera_motion,
        pose_missing_reason=captured.pose_missing_reason,
        backend_received_at_ms=received_ms,
    )


def _map_homography_status(status: str) -> HomographyStatus:
    try:
        return HomographyStatus(status)
    except ValueError:
        return HomographyStatus.UNAVAILABLE


def _draw_player_overlay(
    frame: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    track_id: int,
    entity_id: int | None = None,
    team_label: str,
    role_label: str = "UNKNOWN",
    color: tuple[int, int, int],
) -> None:
    """Draw a compact team marker for the final output video.

    The previous full bounding-box HUD was useful for debugging, but it
    obscured players in the final recording.  The production-style overlay
    uses a small team-coloured ellipse at the player's feet and a compact
    label:

    ``H10`` / ``A39`` for outfield players, ``HG`` / ``AG`` for goalkeepers,
    and ``REF`` for referees.  Unknown semantics are intentionally omitted
    from the final video.
    """

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(x1 + 1, min(frame_width - 1, x2))
    y2 = max(y1 + 1, min(frame_height - 1, y2))

    display_id = entity_id if entity_id is not None else track_id
    label = _format_player_overlay_label(
        team_label=team_label,
        role_label=role_label,
        display_id=display_id,
    )
    if label is None:
        return

    marker_color = color
    if role_label.strip().lower() in {"referee", "ref"}:
        marker_color = REFEREE_OVERLAY_COLOR

    # Keep the marker legible for both distant and nearby players without
    # letting it grow into a large bounding-box-like element.  Only draw the
    # lower half of the ellipse so the marker does not cross the player's
    # legs or torso.
    center_x = int(round((x1 + x2) / 2))
    center_y = min(frame_height - 1, y2 + max(1, int(round((y2 - y1) * 0.03))))
    box_width = max(1, x2 - x1)
    radius_x = max(10, min(30, int(round(box_width * 0.65))))
    radius_y = max(4, min(9, int(round(radius_x * 0.27))))
    cv2.ellipse(
        frame,
        (center_x, center_y),
        (radius_x, radius_y),
        0,
        0,
        180,
        marker_color,
        2,
        cv2.LINE_AA,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.38
    text_thickness = 1
    outline_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        text_thickness,
    )
    padding_x = 4
    padding_y = 2
    box_width = text_width + padding_x * 2
    box_height = text_height + baseline + padding_y * 2
    box_left = max(0, min(frame_width - box_width, center_x - box_width // 2))
    box_top = max(0, min(frame_height - box_height, center_y - box_height // 2))
    box_right = min(frame_width - 1, box_left + box_width)
    box_bottom = min(frame_height - 1, box_top + box_height)
    cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), marker_color, -1)
    text_x = box_left + padding_x
    text_y = box_top + padding_y + text_height
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _format_player_overlay_label(
    *,
    team_label: str,
    role_label: str,
    display_id: int,
) -> Optional[str]:
    """Return the compact final-video label, or ``None`` for UNKNOWN."""

    team = str(team_label).strip().lower()
    role = str(role_label).strip().lower().replace(" predicted", "")
    if role in {"unknown", "", "none"} or team == "unknown":
        return None
    if role in {"referee", "ref"}:
        return "REF"
    if team == "home" and role in {"goalkeeper", "gk"}:
        return "HG"
    if team == "away" and role in {"goalkeeper", "gk"}:
        return "AG"
    if role in {"outfield", "player"} and team in {"home", "away"}:
        prefix = "H" if team == "home" else "A"
        return f"{prefix}{int(display_id)}"
    return None


class InferencePipeline:
    def __init__(
        self,
        source: VideoSource,
        store: StateStore,
        device: str = "cpu",
        mode: PipelineMode = PipelineMode.REALTIME,
        player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
        pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
        camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
        enable_undistortion: bool = True,
        calibration_alpha: float = 0.0,
        pitch_detection_interval: int = 5,
        imgsz: int = 640,
        role_model_path: str = ROLE_DETECTION_MODEL_PATH,
        team_classifier_path: Optional[str] = TEAM_CLASSIFIER_PATH,
        team_calibration_path: Optional[str] = None,
        role_detection_interval: int = 3,
        team_classification_interval: int = 5,
        player_confidence: float = 0.25,
        player_iou: float = 0.7,
        track_activation_threshold: float = 0.25,
        track_lost_buffer: int = 45,
        track_matching_threshold: float = 0.8,
        track_minimum_consecutive_frames: int = 2,
        max_prediction_gap_frames: int = 6,
        track_reactivation_window_frames: int = 12,
        semantic_manager: Optional[object] = None,
        team_assignment_service: Optional[object] = None,
        inference_backend: str = "auto",
        enable_foul_detection: bool = False,
        foul_checkpoint_path: Optional[str] = None,
        foul_confidence_threshold: float = 0.48,
        foul_detector: Optional[object] = None,
        enable_recording: bool = False,
        target_video_path: Optional[str] = None,
        frame_sink: Optional[Callable[[np.ndarray, FrameState], None]] = None,
        frame_observer: Optional[Callable[[np.ndarray, FrameState], None]] = None,
    ) -> None:
        self._source = source
        self._store = store
        self._device = device
        self._mode = mode
        self._player_model_path = player_model_path
        self._pitch_model_path = pitch_model_path
        self._camera_calibration_path = camera_calibration_path
        self._enable_undistortion = enable_undistortion
        self._calibration_alpha = calibration_alpha
        self._pitch_detection_interval = pitch_detection_interval
        self._imgsz = imgsz
        self._role_model_path = role_model_path
        self._team_classifier_path = team_classifier_path
        self._team_calibration_path = team_calibration_path
        self._role_detection_interval = role_detection_interval
        self._team_classification_interval = team_classification_interval
        self._player_confidence = min(max(float(player_confidence), 0.0), 1.0)
        self._player_iou = min(max(float(player_iou), 0.0), 1.0)
        self._max_prediction_gap_frames = max(int(max_prediction_gap_frames), 0)
        self._track_reactivation_window_frames = max(
            int(track_reactivation_window_frames), self._max_prediction_gap_frames
        )
        self._track_activation_threshold = float(track_activation_threshold)
        self._track_lost_buffer = max(int(track_lost_buffer), 1)
        self._track_matching_threshold = float(track_matching_threshold)
        self._track_minimum_consecutive_frames = max(int(track_minimum_consecutive_frames), 1)
        self._inference_backend = inference_backend
        self._enable_foul_detection = enable_foul_detection
        self._foul_checkpoint_path = foul_checkpoint_path or FOUL_MODEL_PATH
        self._foul_confidence_threshold = foul_confidence_threshold
        self._recorder: Optional[VideoRecorder] = None
        if enable_recording:
            recording_path = target_video_path or self._default_recording_path()
            self._recorder = VideoRecorder(recording_path, store, fps=source.fps)
        self._frame_sink = frame_sink
        self._frame_observer = frame_observer
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._buffer = BoundedFrameBuffer(maxsize=4, mode=mode)
        self._dropped_frames = 0

        self._metrics_start = 0.0
        self._metrics_frames = 0
        self._metrics_last = 0.0
        self._last_metrics_emit = 0.0
        self._latency_acc_ms = 0.0
        self._end_to_end_acc_ms = 0.0

        if device.startswith("cuda"):
            self._use_fp16 = torch.cuda.is_available()
        else:
            self._use_fp16 = False

        self._vision_core: Optional[VisionCore] = None
        self._semantic_manager = semantic_manager
        self._team_assignment_service = team_assignment_service
        self._semantic_interval = max(
            1, min(int(role_detection_interval), int(team_classification_interval))
        )
        self._semantic_last_frame: Optional[int] = None
        self._semantic_results: dict[int, object] = {}
        self._display_smoother = TrackDisplaySmoother(
            ema_alpha=0.65,
            max_missing_frames=self._max_prediction_gap_frames,
        )
        self.semantic_inference_count = 0
        self.team_inference_count = 0
        self.team_unknown_count = 0
        self._event_engine = EventEngine()
        self._foul_detector = foul_detector
        self._foul_adapter = FoulEventAdapter(confidence_threshold=foul_confidence_threshold)
        self.foul_inference_count = 0

    def start(self) -> None:
        if self._running:
            return
        self._load_models()
        if self._recorder is not None:
            self._recorder.start()
        self._running = True
        self._store.pipeline_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pipeline")
        self._thread.start()
        logger.info("InferencePipeline started (device=%s, fp16=%s)", self._device, self._use_fp16)

    def stop(self) -> None:
        self._running = False
        self._store.pipeline_running = False
        self._buffer.close()
        # Release every source, including a field source whose queue already
        # reached EOF.  FieldIngestSource uses release() to detach the epoch
        # and stop the receiver; checking is_opened() first would leak it.
        self._source.release()
        recorder = getattr(self, "_recorder", None)
        if recorder is not None:
            recorder.stop()
        logger.info("InferencePipeline stopped")

    @staticmethod
    def _default_recording_path() -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return str(Path("debug") / "recordings" / f"analysis_{timestamp}.mp4")

    @property
    def recording_path(self) -> Optional[str]:
        recorder = getattr(self, "_recorder", None)
        return recorder.target_path if recorder is not None else None

    @property
    def recording_status(self) -> dict[str, object]:
        recorder = getattr(self, "_recorder", None)
        return {
            "enabled": recorder is not None,
            "active": recorder.active if recorder is not None else False,
            "path": self.recording_path,
            "frames_written": recorder.frames_written if recorder is not None else 0,
        }

    def _load_models(self) -> None:
        logger.info("Loading shared vision core")
        self._vision_core = VisionCore(
            device=self._device,
            fps=max(self._source.fps, 1.0),
            player_model_path=self._player_model_path,
            pitch_model_path=self._pitch_model_path,
            camera_calibration_path=self._camera_calibration_path,
            enable_undistortion=self._enable_undistortion,
            calibration_alpha=self._calibration_alpha,
            pitch_detection_interval=self._pitch_detection_interval,
            imgsz=self._imgsz,
            player_confidence=self._player_confidence,
            player_iou=self._player_iou,
            track_activation_threshold=self._track_activation_threshold,
            track_lost_buffer=self._track_lost_buffer,
            track_matching_threshold=self._track_matching_threshold,
            track_minimum_consecutive_frames=self._track_minimum_consecutive_frames,
            max_prediction_gap_frames=self._max_prediction_gap_frames,
            reactivation_window_frames=self._track_reactivation_window_frames,
            inference_backend=self._inference_backend,
        )
        self._vision_core.load_models()
        if self._semantic_manager is None:
            from app.classification.team_calibration.appearance_features import (
                AppearanceFeatureExtractor,
            )
            from app.classification.team_calibration.bundle import CalibrationBundle
            from app.classification.team_calibration.predictor import SupervisedPrototypeClassifier
            from app.classification.team_calibration.role_predictor import CalibratedRoleClassifier
            from app.classification.team_calibration.runtime import TeamAssignmentService
            from app.vision.role import UltralyticsRoleClassifier
            from app.vision.semantics import TrackSemanticManager

            bundle = None
            appearance_extractor = None
            if self._team_calibration_path:
                try:
                    bundle = CalibrationBundle.load(self._team_calibration_path)
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning("Team calibration bundle unavailable; using UNKNOWN: %s", exc)

            if bundle is not None:
                appearance_extractor = AppearanceFeatureExtractor(
                    device=self._device,
                    pretrained=True,
                )

            external_role_classifier = UltralyticsRoleClassifier(
                model_path=self._role_model_path,
                device=self._device,
                imgsz=self._imgsz,
            )
            if not external_role_classifier.load():
                logger.warning(
                    "Role model unavailable at %s; using supervised role prototypes when available",
                    self._role_model_path,
                )
                external_role_classifier = None

            team_classifier = self._team_assignment_service
            if team_classifier is None and bundle is not None:
                team_classifier = TeamAssignmentService(
                    prototypes=bundle.prototypes,
                    classifier=SupervisedPrototypeClassifier(bundle.prototypes),
                    appearance_extractor=appearance_extractor,
                    require_appearance=True,
                )

            role_classifier = None
            if bundle is not None or external_role_classifier is not None:
                role_classifier = CalibratedRoleClassifier(
                    prototypes=bundle.prototypes if bundle is not None else None,
                    model=external_role_classifier,
                    appearance_extractor=appearance_extractor,
                )
            self._semantic_manager = TrackSemanticManager(
                role_classifier=role_classifier,
                team_classifier=team_classifier,
            )
        if self._enable_foul_detection and self._foul_detector is None:
            try:
                from app.foul_detection.detector import FoulDetector

                self._foul_detector = FoulDetector(
                    checkpoint_path=self._foul_checkpoint_path,
                    device=self._device,
                )
            except Exception as exc:
                logger.warning("Foul detector unavailable; continuing without foul events: %s", exc)
                self._foul_detector = None

    def _run_loop(self) -> None:
        self._metrics_start = time.monotonic()
        self._metrics_last = self._metrics_start
        self._last_metrics_emit = self._metrics_start

        try:
            while self._running:
                ret, captured = self._source.read_packet()

                if not ret or captured is None:
                    if not self._source.is_opened():
                        logger.info("Video source ended")
                        self._store.source_status = SourceStatus.DISCONNECTED
                        break
                    time.sleep(0.01)
                    continue

                frame = captured.image
                capture_ts = (
                    captured.capture_unix_us / 1000.0
                    if captured.capture_unix_us is not None
                    else self._source.capture_timestamp_ms()
                )

                inference_start = time.monotonic()
                frame_state = self._process_frame(frame, capture_ts, captured)
                inference_end = time.monotonic()

                if frame_state is not None:
                    inference_latency = (inference_end - inference_start) * 1000
                    frame_state.processed_timestamp_ms = time.time() * 1000
                    end_to_end = frame_state.processed_timestamp_ms - capture_ts

                    self._metrics_frames += 1
                    self._latency_acc_ms += inference_latency
                    self._end_to_end_acc_ms += end_to_end

                    self._store.latest_frame_state = frame_state

                now = time.monotonic()
                if now - self._last_metrics_emit >= METRICS_INTERVAL_SEC:
                    self._emit_metrics()
                    self._last_metrics_emit = now
        finally:
            # A local file can finish without an explicit stop() call.  Reflect
            # that terminal state, release the source, and finalize the
            # optional debug video.  release() is idempotent for live sources.
            self._running = False
            self._store.pipeline_running = False
            self._source.release()
            recorder = getattr(self, "_recorder", None)
            if recorder is not None:
                recorder.stop()

    def _process_frame(
        self,
        frame: np.ndarray,
        capture_timestamp_ms: float,
        captured: CapturedFrame | None = None,
    ) -> Optional[FrameState]:
        if self._vision_core is None:
            raise RuntimeError("VisionCore is not loaded")

        with torch.inference_mode():
            vision_frame = self._vision_core.process(frame, self._source.frame_count)

        detections = vision_frame.tracked_detections
        projection = vision_frame.projection

        foul_event = self._process_foul(
            frame=vision_frame.undistorted_frame,
            frame_id=self._source.frame_count,
            timestamp_s=capture_timestamp_ms / 1000.0,
        )

        player_states: list[PlayerState] = []
        semantic_results = self._update_semantics(
            rebindings=vision_frame.rebindings,
            frame=vision_frame.undistorted_frame,
            detections=detections,
            frame_index=self._source.frame_count,
        )
        for idx in range(len(detections)):
            tracker_id = (
                int(detections.tracker_id[idx]) if detections.tracker_id is not None else idx
            )
            entity_id = vision_frame.entity_ids.get(tracker_id, tracker_id)
            track_status = vision_frame.track_status.get(tracker_id, "detected")
            field_xy = vision_frame.field_xy[idx]
            has_field_xy = bool(np.isfinite(field_xy).all())
            semantic = semantic_results.get(tracker_id)
            role = PlayerRole.UNKNOWN
            team = TeamLabel.UNKNOWN
            team_id = -1
            role_confidence = 0.0
            team_confidence = 0.0
            team_rejection_reason = None
            semantic_status = "unknown"
            if semantic is not None:
                try:
                    role_value = getattr(semantic, "role", "unknown")
                    normalized_role = str(getattr(role_value, "value", role_value))
                    if normalized_role == "player":
                        normalized_role = PlayerRole.OUTFIELD.value
                    role = PlayerRole(normalized_role)
                except ValueError:
                    role = PlayerRole.UNKNOWN
                raw_team = getattr(semantic, "team", TeamLabel.UNKNOWN)
                try:
                    team = TeamLabel(str(getattr(raw_team, "value", raw_team)))
                except ValueError:
                    team = TeamLabel.UNKNOWN
                try:
                    candidate_team = int(getattr(semantic, "team_id", -1))
                    team_id = candidate_team if candidate_team in (0, 1) else -1
                except (TypeError, ValueError):
                    team_id = -1
                if team == TeamLabel.UNKNOWN and team_id in (0, 1):
                    team = TeamLabel.HOME if team_id == 0 else TeamLabel.AWAY
                if team in {TeamLabel.HOME, TeamLabel.AWAY}:
                    team_id = 0 if team == TeamLabel.HOME else 1
                role_confidence = float(getattr(semantic, "role_confidence", 0.0))
                team_confidence = float(getattr(semantic, "team_confidence", 0.0))
                team_rejection_reason = getattr(semantic, "team_rejection_reason", None)
                semantic_status = str(
                    getattr(semantic, "semantic_status", getattr(semantic, "status", "unknown"))
                )
            player_states.append(
                PlayerState(
                    track_id=tracker_id,
                    entity_id=entity_id,
                    track_status=track_status,
                    missing_frames=0,
                    role=role,
                    team=team,
                    team_label=team,
                    team_id=team_id,
                    field_x=float(field_xy[0]) if has_field_xy else None,
                    field_y=float(field_xy[1]) if has_field_xy else None,
                    confidence=(
                        float(detections.confidence[idx])
                        if detections.confidence is not None
                        else 0.0
                    ),
                    role_confidence=role_confidence,
                    team_confidence=team_confidence,
                    team_rejection_reason=team_rejection_reason,
                    bbox=tuple(float(value) for value in detections.xyxy[idx]),
                    semantic_status=semantic_status,
                )
            )
            self.team_inference_count = getattr(self, "team_inference_count", 0) + 1
            if team_id == -1:
                self.team_unknown_count = getattr(self, "team_unknown_count", 0) + 1

        # Draw smoothed display boxes. Brief detector gaps are held only for
        # visualization; stale tracks never enter FrameState or classifiers.
        annotated_frame = vision_frame.undistorted_frame.copy()
        display_smoother = getattr(self, "_display_smoother", None)
        if display_smoother is None:
            # Keep lightweight ``__new__``-constructed test doubles and legacy
            # callers compatible with the new display-only state.
            display_smoother = TrackDisplaySmoother(
                ema_alpha=0.65,
                max_missing_frames=getattr(self, "_max_prediction_gap_frames", 6),
            )
            self._display_smoother = display_smoother
        for display in display_smoother.update(player_states):
            x1, y1, x2, y2 = map(int, display.bbox)
            color = {0: HOME_OVERLAY_COLOR, 1: AWAY_OVERLAY_COLOR}.get(
                display.team_id,
                UNKNOWN_OVERLAY_COLOR,
            )
            _draw_player_overlay(
                annotated_frame,
                bbox=(x1, y1, x2, y2),
                track_id=display.track_id,
                entity_id=display.entity_id,
                team_label=display.team_label.upper(),
                role_label=display.role_label.upper()
                + (" PREDICTED" if display.track_status == "predicted" else ""),
                color=color,
            )

        publish_frame = getattr(self._store, "publish_raw_frame", None)
        if publish_frame is not None:
            publish_frame(annotated_frame)
        else:
            self._store._latest_raw_frame = annotated_frame  # type: ignore[attr-defined]

        recorder = getattr(self, "_recorder", None)
        if recorder is not None:
            recorder.write(annotated_frame)

        elapsed = time.monotonic() - self._metrics_start
        current_fps = self._metrics_frames / max(elapsed, 0.001)

        frame_state = FrameState(
            frame_id=self._source.frame_count,
            capture_timestamp_ms=capture_timestamp_ms,
            processing_fps=current_fps,
            homography_status=_map_homography_status(projection.homography_status),
            players=player_states,
            events=[],
            capture_source=_capture_source_metadata(captured),
        )
        event_engine = getattr(self, "_event_engine", None)
        if event_engine is not None:
            frame_state.events = event_engine.update(frame_state)
        if foul_event is not None:
            frame_state.events.append(foul_event)
        for event in frame_state.events:
            self._store.add_event(event)

        frame_observer = getattr(self, "_frame_observer", None)
        if frame_observer is not None:
            try:
                frame_observer(vision_frame.undistorted_frame, frame_state)
            except Exception as exc:
                logger.warning("Frame observer failed; continuing pipeline: %s", exc)

        frame_sink = getattr(self, "_frame_sink", None)
        if frame_sink is not None:
            try:
                frame_sink(annotated_frame, frame_state)
            except Exception as exc:
                # Rendering is an optional observability feature.  A failed
                # sink must not terminate the inference worker or the API
                # stream.
                logger.warning("Frame sink failed; continuing pipeline: %s", exc)
        return frame_state

    def _process_foul(
        self,
        *,
        frame: np.ndarray,
        frame_id: int,
        timestamp_s: float,
    ):
        foul_detector = getattr(self, "_foul_detector", None)
        if foul_detector is None:
            return None
        try:
            prediction = foul_detector.update(frame)
            self.foul_inference_count += 1
            foul_adapter = getattr(self, "_foul_adapter", FoulEventAdapter())
            return foul_adapter.update(
                prediction,
                frame_id=frame_id,
                timestamp=timestamp_s,
                field_xy=None,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Foul prediction failed; skipping event: %s", exc)
            return None

    def _update_semantics(
        self,
        frame: np.ndarray,
        detections: Any,
        frame_index: int,
        rebindings: Optional[dict[int, int]] = None,
    ) -> dict[int, object]:
        if self._semantic_manager is None:
            return {}
        rebind = getattr(self._semantic_manager, "rebind_track", None)
        if callable(rebind):
            for new_track_id, old_track_id in (rebindings or {}).items():
                try:
                    rebind(new_track_id, old_track_id)
                except (AttributeError, TypeError, ValueError) as exc:
                    logger.debug("Semantic rebind skipped: %s", exc)
        if (
            self._semantic_last_frame is None
            or frame_index - self._semantic_last_frame >= self._semantic_interval
        ):
            try:
                self._semantic_results = self._semantic_manager.update(
                    frame=frame,
                    detections=detections,
                    frame_index=frame_index,
                )
                self.semantic_inference_count += 1
                self._semantic_last_frame = frame_index
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Semantic prediction failed; using previous state: %s", exc)
        current_ids = (
            {int(track_id) for track_id in detections.tracker_id}
            if detections.tracker_id is not None
            else set()
        )
        return {
            track_id: result
            for track_id, result in self._semantic_results.items()
            if track_id in current_ids
        }

    def _emit_metrics(self) -> None:
        elapsed = time.monotonic() - self._metrics_start
        fps = self._metrics_frames / max(elapsed, 0.001)
        avg_lat = self._latency_acc_ms / max(self._metrics_frames, 1)
        avg_e2e = self._end_to_end_acc_ms / max(self._metrics_frames, 1)

        memory_mb = 0.0
        try:
            import psutil

            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        gpu_mem_mb = None
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_mb = mem_info.used / (1024 * 1024)
        except Exception:
            pass

        vision_core = self._vision_core
        processed = vision_core.frames_processed if vision_core is not None else 0
        pitch_calls = vision_core.pitch_detection_count if vision_core is not None else 0
        reused = vision_core.pitch_reuse_count if vision_core is not None else 0
        available = vision_core.homography_available_count if vision_core is not None else 0
        camera_motion_refreshes = (
            vision_core.camera_motion_refresh_count if vision_core is not None else 0
        )
        player_calls = vision_core.player_inference_count if vision_core is not None else 0
        player_time = vision_core.player_inference_time_ms if vision_core is not None else 0.0
        pitch_time = vision_core.pitch_inference_time_ms if vision_core is not None else 0.0
        track_interruptions = vision_core.track_id_interruptions if vision_core is not None else 0
        track_occlusion_events = (
            vision_core.track_occlusion_events if vision_core is not None else 0
        )
        track_predicted_frames = (
            vision_core.track_predicted_frames if vision_core is not None else 0
        )
        track_reactivated_count = (
            vision_core.track_reactivated_count if vision_core is not None else 0
        )
        track_id_switches = vision_core.track_id_switches if vision_core is not None else 0
        track_recovered_count = vision_core.track_recovered_count if vision_core is not None else 0
        track_fragmentations = vision_core.track_fragmentations if vision_core is not None else 0
        track_max_missing_frames = (
            vision_core.track_max_missing_frames if vision_core is not None else 0
        )
        track_entity_rebinds = vision_core.track_entity_rebinds if vision_core is not None else 0
        track_entity_fragmentations = (
            vision_core.track_entity_fragmentations if vision_core is not None else 0
        )
        track_lifecycle_counts = (
            dict(vision_core.track_lifecycle_counts) if vision_core is not None else {}
        )
        semantic_switches = (
            int(getattr(self._semantic_manager, "semantic_label_switches", 0))
            if self._semantic_manager is not None
            else 0
        )
        team_switches = (
            int(getattr(self._semantic_manager, "team_label_switches", 0))
            if self._semantic_manager is not None
            else 0
        )
        stream_metrics = getattr(self._source, "stream_metrics", {})
        field_session_id = stream_metrics.get("session_id")
        field_stream_epoch = stream_metrics.get("stream_epoch")

        metrics = MetricsSnapshot(
            processing_fps=round(fps, 1),
            input_fps=round(self._source.fps, 1),
            inference_latency_ms=round(avg_lat, 1),
            end_to_end_latency_ms=round(avg_e2e, 1),
            dropped_frames=self._buffer.dropped_frames,
            queue_length=len(self._buffer),
            player_count=len(self._store.latest_frame_state.players)
            if self._store.latest_frame_state
            else 0,
            source_status=self._store.source_status,
            memory_mb=round(memory_mb, 1),
            gpu_memory_mb=round(gpu_mem_mb, 1) if gpu_mem_mb is not None else None,
            player_inference_latency_ms=round(player_time / max(player_calls, 1), 1),
            pitch_inference_latency_ms=round(pitch_time / max(pitch_calls, 1), 1),
            pitch_detection_count=pitch_calls,
            homography_reuse_ratio=round(reused / max(processed, 1), 3),
            homography_available_ratio=round(available / max(processed, 1), 3),
            camera_motion_refresh_count=camera_motion_refreshes,
            track_id_interruptions=track_interruptions,
            track_occlusion_events=track_occlusion_events,
            track_predicted_frames=track_predicted_frames,
            track_recovered_count=track_recovered_count,
            track_reactivated_count=track_reactivated_count,
            track_id_switches=track_id_switches,
            track_fragmentations=track_fragmentations,
            track_max_missing_frames=track_max_missing_frames,
            track_entity_rebinds=track_entity_rebinds,
            track_entity_fragmentations=track_entity_fragmentations,
            track_lifecycle_counts=track_lifecycle_counts,
            semantic_inference_count=self.semantic_inference_count,
            semantic_label_switches=semantic_switches,
            team_inference_count=getattr(self, "team_inference_count", 0),
            team_unknown_rate=round(
                getattr(self, "team_unknown_count", 0)
                / max(getattr(self, "team_inference_count", 0), 1),
            ),
            team_label_switches=team_switches,
            jpeg_frames_encoded=int(getattr(self._store, "jpeg_frames_encoded", 0)),
            jpeg_encode_latency_ms=round(
                float(getattr(self._store, "jpeg_encode_latency_ms", 0.0)), 3
            ),
            foul_inference_count=self.foul_inference_count,
            decoded_frames=int(stream_metrics.get("decoded_frame_count", 0)),
            decode_dropped_frames=int(stream_metrics.get("decode_dropped_frames", 0)),
            join_missing_frames=int(stream_metrics.get("join_missing_count", 0)),
            inference_dropped_frames=self._buffer.dropped_frames,
            field_session_id=field_session_id,
            field_stream_epoch=field_stream_epoch,
        )
        self._store.metrics = metrics


def detection_confidence(detection: Any) -> float:
    try:
        if hasattr(detection, "confidence") and len(detection) > 2:
            return float(detection.confidence[0]) if detection.confidence is not None else 0.0
        return 0.0
    except (IndexError, TypeError):
        return 0.0
