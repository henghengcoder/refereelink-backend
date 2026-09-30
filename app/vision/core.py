"""Shared person detection, ByteTrack and pitch-coordinate pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch

from app.config.pitch import SoccerPitchConfiguration
from app.constants.classes import UNKNOWN_COLOR_ID
from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
)
from app.geometry.camera import (
    CameraMotionEstimate,
    CameraMotionEstimator,
    CameraUndistorter,
    build_undistorter,
)
from app.geometry.pitch_projection import (
    PitchProjectionEngine,
    PitchProjectionResult,
    blend_homographies,
    homography_deviation,
    translate_homography,
)
from app.geometry.position_filter import PlayerPositionFilter
from app.vision.entities import TrackEntityManager

# Weight given to a newly fitted homography when the camera has not moved.
# A fit from the bare minimum of four keypoints has no redundancy, so trust it
# less; the blend removes the whole-pitch "jump" at every scheduled refresh.
MINIMAL_FIT_BLEND_ALPHA = 0.3
REDUNDANT_FIT_BLEND_ALPHA = 0.6
# A new fit that moves the players' feet by more than this (median) compared
# with the matrix on screen is treated as a bad keypoint fit and ignored, up to
# a few consecutive times so a genuinely changed view is still accepted.
HOMOGRAPHY_REJECT_DEVIATION_CM = 200.0
MAX_CONSECUTIVE_HOMOGRAPHY_REJECTS = 6
# Motion-triggered refreshes are compared with a translation-compensated
# matrix, which drifts during long pans, so they get a looser, shorter gate.
MOTION_REFRESH_REJECT_DEVIATION_CM = 400.0
MAX_CONSECUTIVE_MOTION_REFRESH_REJECTS = 3


def _empty_detections() -> sv.Detections:
    return sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))


class TrackLifecycleState(str, Enum):
    """Observable lifecycle states for a raw tracker ID."""

    DETECTED = "detected"
    PREDICTED = "predicted"
    OCCLUDED = "occluded"
    LOST = "lost"
    REACTIVATED = "reactivated"
    ID_SWITCH = "id_switch"


@dataclass
class VisionFrame:
    undistorted_frame: np.ndarray
    detections: sv.Detections
    tracked_detections: sv.Detections
    projection: PitchProjectionResult
    field_xy: np.ndarray
    color_lookup: np.ndarray
    person_only: bool
    entity_ids: dict[int, int] = field(default_factory=dict)
    track_status: dict[int, str] = field(default_factory=dict)
    rebindings: dict[int, int] = field(default_factory=dict)
    lifecycle_events: tuple[dict[str, object], ...] = ()

    @property
    def homography_status(self) -> str:
        return self.projection.homography_status


class VisionCore:
    """One reusable detection/tracking/projection implementation.

    The default player model is an official YOLOv11 COCO checkpoint and is
    therefore intentionally treated as a person-only detector.  Role-aware
    custom checkpoints can be enabled later without changing the output
    shape by setting ``person_only=False``.
    """

    def __init__(
        self,
        device: str = "cpu",
        fps: float = 25.0,
        player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
        pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
        camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
        enable_undistortion: bool = True,
        calibration_alpha: float = 0.0,
        pitch_detection_interval: int = 5,
        imgsz: int = 640,
        enable_player: bool = True,
        enable_pitch: bool = True,
        person_only: bool = True,
        undistorter: Optional[CameraUndistorter] = None,
        player_model: object = None,
        pitch_model: object = None,
        tracker: Optional[object] = None,
        projection_engine: Optional[PitchProjectionEngine] = None,
        inference_backend: str = "auto",
        camera_motion_threshold_px: float = 6.0,
        camera_motion_estimator: Optional[CameraMotionEstimator] = None,
        player_confidence: float = 0.25,
        player_iou: float = 0.7,
        track_activation_threshold: float = 0.25,
        track_lost_buffer: int = 45,
        track_matching_threshold: float = 0.8,
        track_minimum_consecutive_frames: int = 2,
        max_prediction_gap_frames: int = 6,
        reactivation_window_frames: int = 12,
        entity_manager: Optional[TrackEntityManager] = None,
        stabilize_projection: bool = True,
        position_filter: Optional[PlayerPositionFilter] = None,
    ) -> None:
        self.device = device
        self.fps = max(float(fps), 1.0)
        self.player_model_path = player_model_path
        self.pitch_model_path = pitch_model_path
        self.pitch_detection_interval = max(int(pitch_detection_interval), 1)
        self.imgsz = max(int(imgsz), 32)
        self.player_confidence = min(max(float(player_confidence), 0.0), 1.0)
        self.player_iou = min(max(float(player_iou), 0.0), 1.0)
        self.max_prediction_gap_frames = max(int(max_prediction_gap_frames), 0)
        self._entity_manager = entity_manager or TrackEntityManager(
            max_prediction_gap_frames=self.max_prediction_gap_frames,
            reactivation_window_frames=max(
                int(reactivation_window_frames), self.max_prediction_gap_frames
            ),
        )
        self.track_lost_buffer = max(int(track_lost_buffer), 1)
        self.enable_player = enable_player
        self.enable_pitch = enable_pitch
        self.person_only = person_only
        self.inference_backend = inference_backend
        self.camera_motion_refresh_count = 0
        self._player_model = player_model
        self._pitch_model = pitch_model
        self._tracker = (
            tracker
            if tracker is not None
            else sv.ByteTrack(
                track_activation_threshold=float(track_activation_threshold),
                lost_track_buffer=self.track_lost_buffer,
                minimum_matching_threshold=float(track_matching_threshold),
                frame_rate=self.fps,
                minimum_consecutive_frames=max(int(track_minimum_consecutive_frames), 1),
            )
        )
        self._projection_engine = (
            projection_engine
            if projection_engine is not None
            else PitchProjectionEngine(config=SoccerPitchConfiguration(), fps=self.fps)
        )
        self._last_pitch_detection_frame: Optional[int] = None
        self._use_fp16 = device.startswith("cuda") and torch.cuda.is_available()
        self.frames_processed = 0
        self.player_inference_count = 0
        self.pitch_detection_count = 0
        self.pitch_reuse_count = 0
        self.homography_available_count = 0
        self.track_id_interruptions = 0
        self.track_occlusion_events = 0
        self.track_predicted_frames = 0
        self.track_recovered_count = 0
        self.track_reactivated_count = 0
        self.track_id_switches = 0
        self.track_fragmentations = 0
        self.track_max_missing_frames = 0
        self.track_entity_rebinds = 0
        self.track_entity_fragmentations = 0
        self._previous_track_ids: set[int] = set()
        self._track_missing_frames: dict[int, int] = {}
        self._track_last_lifecycle_state: dict[int, str] = {}
        self.track_lifecycle: dict[int, list[str]] = {}
        self.track_lifecycle_events: list[dict[str, object]] = []
        self.track_lifecycle_counts: dict[str, int] = {
            state.value: 0 for state in TrackLifecycleState
        }
        self.player_inference_time_ms = 0.0
        self.pitch_inference_time_ms = 0.0
        self._undistorter = undistorter or build_undistorter(
            calibration_path=camera_calibration_path,
            enabled=enable_undistortion,
            alpha=calibration_alpha,
        )
        self._camera_motion_estimator = camera_motion_estimator or CameraMotionEstimator(
            threshold_px=camera_motion_threshold_px
        )
        # Temporal stabilisation of the pitch projection: homography blending,
        # sub-threshold camera-motion compensation and per-player filtering.
        self.stabilize_projection = bool(stabilize_projection)
        self._position_filter = position_filter or PlayerPositionFilter(fps=self.fps)
        self._base_homography: Optional[np.ndarray] = None
        self._last_output_homography: Optional[np.ndarray] = None
        self._consecutive_homography_rejects = 0
        self.homography_rejections = 0

    @property
    def projection_engine(self) -> PitchProjectionEngine:
        return self._projection_engine

    def load_models(self) -> None:
        from app.vision.backends import UltralyticsBackend

        if self.enable_player and self._player_model is None:
            backend = None if self.inference_backend == "auto" else self.inference_backend
            self._player_model = UltralyticsBackend(
                self.player_model_path,
                backend=backend,
                device=self.device,
            )
        if self.enable_pitch and self._pitch_model is None:
            backend = None if self.inference_backend == "auto" else self.inference_backend
            self._pitch_model = UltralyticsBackend(
                self.pitch_model_path,
                backend=backend,
                device=self.device,
            )

    def _run_model(
        self,
        model: object,
        frame: np.ndarray,
        *,
        conf: float | None = None,
        iou: float | None = None,
    ):
        kwargs = {"imgsz": self.imgsz, "half": self._use_fp16}
        if conf is not None:
            kwargs["conf"] = conf
        if iou is not None:
            kwargs["iou"] = iou
        if hasattr(model, "predict"):
            try:
                return model.predict(frame, **kwargs)
            except TypeError:
                try:
                    return model.predict(frame, imgsz=self.imgsz)
                except TypeError:
                    return model.predict(frame)
        try:
            return model(frame, verbose=False, **kwargs)
        except TypeError:
            try:
                return model(frame, imgsz=self.imgsz, verbose=False)
            except TypeError:
                return model(frame)

    def _predict_player(self, frame: np.ndarray) -> sv.Detections:
        if not self.enable_player:
            return _empty_detections()
        if self._player_model is None:
            raise RuntimeError("Player model has not been loaded")
        result = self._run_model(
            self._player_model,
            frame,
            conf=self.player_confidence,
            iou=self.player_iou,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        if not self.person_only or len(detections) == 0:
            return detections
        if detections.class_id is None:
            return detections
        return detections[detections.class_id == 0]

    def _predict_pitch(self, frame: np.ndarray) -> sv.KeyPoints:
        if self._pitch_model is None:
            raise RuntimeError("Pitch model has not been loaded")
        result = self._run_model(self._pitch_model, frame)[0]
        return sv.KeyPoints.from_ultralytics(result)

    def _should_detect_pitch(self, frame_index: int) -> bool:
        return (
            self._last_pitch_detection_frame is None
            or frame_index - self._last_pitch_detection_frame
            >= self.pitch_detection_interval
        )

    def _projection_for_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        force_refresh: bool = False,
    ) -> PitchProjectionResult:
        if not self.enable_pitch:
            return PitchProjectionResult(
                tracking_observations=[],
                projected_keypoints=[],
                homography=None,
                homography_status="unavailable",
                reprojection_error=None,
            )
        if force_refresh or self._should_detect_pitch(frame_index):
            start = time.perf_counter()
            keypoints = self._predict_pitch(frame)
            self.pitch_inference_time_ms += (time.perf_counter() - start) * 1000
            self.pitch_detection_count += 1
            self._last_pitch_detection_frame = frame_index
            projection = self._projection_engine.update(frame=frame, keypoints=keypoints)
            if force_refresh and projection.homography_status != "fresh":
                invalidate = getattr(self._projection_engine, "invalidate", None)
                if callable(invalidate):
                    invalidate()
                return PitchProjectionResult(
                    tracking_observations=[],
                    projected_keypoints=[],
                    homography=None,
                    homography_status="unavailable",
                    reprojection_error=None,
                )
            return projection
        self.pitch_reuse_count += 1
        return self._projection_engine.reuse(frame=frame)

    def _stabilize_homography(
        self,
        projection: PitchProjectionResult,
        *,
        motion: Optional[CameraMotionEstimate],
        force_refresh: bool,
        frame_shape: tuple[int, ...],
        anchor_points: np.ndarray,
    ) -> PitchProjectionResult:
        """Smooth homography refreshes and compensate small camera drift.

        * A new fit that would move the players' feet (``anchor_points``) by
          more than the rejection threshold is treated as a bad keypoint fit
          and ignored, at most a few times in a row.
        * A scheduled (non motion-triggered) refresh is blended with the matrix
          shown on the previous frame instead of replacing it outright.
        * Frames that reuse the last fit apply the measured image translation
          since that fit, so sub-threshold pans do not accumulate error until
          the next refresh.
        """

        if projection.homography is None:
            self._base_homography = None
            self._last_output_homography = None
            self._consecutive_homography_rejects = 0
            return projection

        if projection.homography_status == "fresh":
            homography = np.asarray(projection.homography, dtype=np.float64)
            previous = self._last_output_homography
            deviation = (
                homography_deviation(previous, homography, anchor_points)
                if previous is not None
                else None
            )
            if force_refresh:
                reject_threshold = MOTION_REFRESH_REJECT_DEVIATION_CM
                max_rejects = MAX_CONSECUTIVE_MOTION_REFRESH_REJECTS
            else:
                reject_threshold = HOMOGRAPHY_REJECT_DEVIATION_CM
                max_rejects = MAX_CONSECUTIVE_HOMOGRAPHY_REJECTS
            if (
                deviation is not None
                and deviation > reject_threshold
                and self._consecutive_homography_rejects < max_rejects
            ):
                # Keep the on-screen matrix.  The motion reference was just
                # re-marked on this frame, so it becomes the new base.
                self._consecutive_homography_rejects += 1
                self.homography_rejections += 1
                self._base_homography = previous
                self._last_output_homography = previous
                return replace(projection, homography=previous)
            # After repeated rejections the view has genuinely changed: adopt
            # the new fit outright instead of blending towards it.
            overridden = self._consecutive_homography_rejects >= max_rejects
            self._consecutive_homography_rejects = 0
            if not force_refresh and not overridden and previous is not None:
                alpha = (
                    REDUNDANT_FIT_BLEND_ALPHA
                    if len(projection.tracking_observations) > 4
                    else MINIMAL_FIT_BLEND_ALPHA
                )
                config = self._projection_engine.config
                homography = blend_homographies(
                    previous,
                    homography,
                    alpha,
                    frame_shape,
                    pitch_length=float(config.length),
                    pitch_width=float(config.width),
                )
            self._base_homography = homography
        elif self._base_homography is not None:
            homography = self._base_homography
            if motion is not None and motion.response >= motion.minimum_response:
                homography = translate_homography(
                    homography, (motion.shift_x_px, motion.shift_y_px)
                )
        else:
            homography = np.asarray(projection.homography, dtype=np.float64)

        self._last_output_homography = homography
        return replace(projection, homography=homography)

    def _field_coordinates(
        self,
        detections: sv.Detections,
        projection: PitchProjectionResult,
    ) -> np.ndarray:
        field_xy = np.full((len(detections), 2), np.nan, dtype=np.float32)
        if projection.homography is None or len(detections) == 0:
            return field_xy

        image_xy = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        ).astype(np.float32)
        try:
            transformed = cv2.perspectiveTransform(
                image_xy.reshape(-1, 1, 2), projection.homography
            ).reshape(-1, 2)
        except cv2.error:
            return field_xy

        config = self._projection_engine.config
        valid = (
            np.isfinite(transformed).all(axis=1)
            & (transformed[:, 0] >= 0)
            & (transformed[:, 0] <= config.length)
            & (transformed[:, 1] >= 0)
            & (transformed[:, 1] <= config.width)
        )
        field_xy[valid] = transformed[valid]
        return field_xy

    def _record_lifecycle(
        self,
        *,
        track_id: int,
        state: TrackLifecycleState,
        frame_index: int,
        entity_id: int | None = None,
        force: bool = False,
    ) -> dict[str, object] | None:
        """Record a lifecycle transition once, except explicitly forced events."""

        state_value = state.value
        previous = self._track_last_lifecycle_state.get(track_id)
        if not force and previous == state_value:
            return None
        event: dict[str, object] = {
            "frame_index": int(frame_index),
            "track_id": int(track_id),
            "entity_id": entity_id,
            "state": state_value,
            "previous_state": previous,
        }
        self.track_lifecycle.setdefault(track_id, []).append(state_value)
        self.track_lifecycle_events.append(event)
        self.track_lifecycle_counts[state_value] = (
            self.track_lifecycle_counts.get(state_value, 0) + 1
        )
        self._track_last_lifecycle_state[track_id] = state_value
        return event

    def process(self, frame: np.ndarray, frame_index: int) -> VisionFrame:
        undistorted_frame = self._undistorter.apply(frame)
        start = time.perf_counter()
        detections = self._predict_player(undistorted_frame)
        self.player_inference_time_ms += (time.perf_counter() - start) * 1000
        self.player_inference_count += 1
        tracked_detections = self._tracker.update_with_detections(detections)
        entity_update = self._entity_manager.update(
            tracked_detections,
            undistorted_frame,
            frame_index,
        )
        self.track_entity_rebinds += len(entity_update.rebindings)
        self.track_entity_fragmentations += entity_update.fragmentations
        current_track_ids = (
            {int(track_id) for track_id in tracked_detections.tracker_id}
            if tracked_detections.tracker_id is not None
            else set()
        )
        known_track_ids = set(self._track_missing_frames) | self._previous_track_ids
        missing_ids = known_track_ids - current_track_ids
        recovered_ids = {
            track_id
            for track_id in current_track_ids
            if self._track_missing_frames.get(track_id, 0) > 0
        }
        newly_missing_ids = {
            track_id
            for track_id in missing_ids
            if self._track_missing_frames.get(track_id, 0) == 0
        }
        if newly_missing_ids:
            self.track_id_interruptions += len(newly_missing_ids)
            self.track_occlusion_events += len(newly_missing_ids)
        if recovered_ids:
            self.track_recovered_count += len(recovered_ids)
        frame_lifecycle_events: list[dict[str, object]] = []
        for track_id in current_track_ids:
            entity_id = entity_update.entity_ids.get(track_id)
            missing_before = self._track_missing_frames.get(track_id, 0)
            if track_id in entity_update.rebindings:
                self.track_id_switches += 1
                switch_event = self._record_lifecycle(
                    track_id=track_id,
                    state=TrackLifecycleState.ID_SWITCH,
                    frame_index=frame_index,
                    entity_id=entity_id,
                    force=True,
                )
                if switch_event is not None:
                    frame_lifecycle_events.append(switch_event)
                self.track_reactivated_count += 1
                reactivated_event = self._record_lifecycle(
                    track_id=track_id,
                    state=TrackLifecycleState.REACTIVATED,
                    frame_index=frame_index,
                    entity_id=entity_id,
                    force=True,
                )
                if reactivated_event is not None:
                    frame_lifecycle_events.append(reactivated_event)
            elif missing_before > 0:
                self.track_reactivated_count += 1
                reactivated_event = self._record_lifecycle(
                    track_id=track_id,
                    state=TrackLifecycleState.REACTIVATED,
                    frame_index=frame_index,
                    entity_id=entity_id,
                )
                if reactivated_event is not None:
                    frame_lifecycle_events.append(reactivated_event)
            else:
                detected_event = self._record_lifecycle(
                    track_id=track_id,
                    state=TrackLifecycleState.DETECTED,
                    frame_index=frame_index,
                    entity_id=entity_id,
                )
                if detected_event is not None:
                    frame_lifecycle_events.append(detected_event)
        for track_id in current_track_ids:
            self._track_missing_frames[track_id] = 0
        for track_id in missing_ids:
            missing = self._track_missing_frames.get(track_id, 0) + 1
            self._track_missing_frames[track_id] = missing
            self.track_max_missing_frames = max(self.track_max_missing_frames, missing)
            if missing == 1 and self.max_prediction_gap_frames > 0:
                occluded_event = self._record_lifecycle(
                    track_id=track_id,
                    state=TrackLifecycleState.OCCLUDED,
                    frame_index=frame_index,
                    entity_id=None,
                )
                if occluded_event is not None:
                    frame_lifecycle_events.append(occluded_event)
            if missing <= self.max_prediction_gap_frames:
                self.track_predicted_frames += 1
                state = TrackLifecycleState.PREDICTED
            elif missing == self.max_prediction_gap_frames + 1:
                self.track_fragmentations += 1
                state = TrackLifecycleState.LOST
            else:
                state = TrackLifecycleState.LOST
            lifecycle_event = self._record_lifecycle(
                track_id=track_id,
                state=state,
                frame_index=frame_index,
                entity_id=None,
            )
            if lifecycle_event is not None:
                frame_lifecycle_events.append(lifecycle_event)
        for track_id in list(self._track_missing_frames):
            if self._track_missing_frames[track_id] > max(
                self.track_lost_buffer,
                self.max_prediction_gap_frames,
            ):
                del self._track_missing_frames[track_id]
        self._previous_track_ids = current_track_ids
        motion = self._camera_motion_estimator.measure(undistorted_frame)
        force_pitch_refresh = bool(motion is not None and motion.requires_refresh)
        if force_pitch_refresh:
            self.camera_motion_refresh_count += 1
        projection = self._projection_for_frame(
            undistorted_frame,
            frame_index,
            force_refresh=force_pitch_refresh,
        )
        if projection.homography_status == "fresh":
            self._camera_motion_estimator.mark_reference(undistorted_frame)
        if self.stabilize_projection:
            projection = self._stabilize_homography(
                projection,
                motion=motion,
                force_refresh=force_pitch_refresh,
                frame_shape=undistorted_frame.shape,
                anchor_points=tracked_detections.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
                if len(tracked_detections)
                else np.empty((0, 2)),
            )
        field_xy = self._field_coordinates(tracked_detections, projection)
        if self.stabilize_projection and tracked_detections.tracker_id is not None:
            keys = [
                entity_update.entity_ids.get(int(track_id), int(track_id))
                for track_id in tracked_detections.tracker_id
            ]
            box_heights = tracked_detections.xyxy[:, 3] - tracked_detections.xyxy[:, 1]
            field_xy = self._position_filter.update(keys, field_xy, frame_index, box_heights)
        self.frames_processed += 1
        if projection.available:
            self.homography_available_count += 1
        color_lookup = np.full(
            len(tracked_detections),
            UNKNOWN_COLOR_ID if self.person_only else 0,
            dtype=np.int64,
        )
        return VisionFrame(
            undistorted_frame=undistorted_frame,
            detections=detections,
            tracked_detections=tracked_detections,
            projection=projection,
            field_xy=field_xy,
            color_lookup=color_lookup,
            person_only=self.person_only,
            entity_ids=entity_update.entity_ids,
            track_status=entity_update.statuses,
            rebindings=entity_update.rebindings,
            lifecycle_events=tuple(frame_lifecycle_events),
        )
