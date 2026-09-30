from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.classification.team_calibration.types import TeamLabel


class HomographyStatus(str, Enum):
    FRESH = "fresh"
    REUSED = "reused"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class BallStatus(str, Enum):
    FRESH = "fresh"
    PREDICTED = "predicted"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class PlayerRole(str, Enum):
    OUTFIELD = "outfield"
    # Backwards-compatible alias for existing event and client code.
    PLAYER = "outfield"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    STAFF = "staff"
    UNKNOWN = "unknown"


class SourceStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class PlayerState(BaseModel):
    track_id: int
    entity_id: Optional[int] = None
    track_status: str = "detected"
    missing_frames: int = 0
    role: PlayerRole
    team: TeamLabel = TeamLabel.UNKNOWN
    team_label: TeamLabel = TeamLabel.UNKNOWN
    team_id: int
    field_x: Optional[float] = None
    field_y: Optional[float] = None
    confidence: float
    role_confidence: float = 0.0
    team_confidence: float = 0.0
    team_rejection_reason: Optional[str] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    semantic_status: str = "unknown"
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None


class BallState(BaseModel):
    status: BallStatus = BallStatus.UNAVAILABLE
    image_x: Optional[float] = None
    image_y: Optional[float] = None
    field_x: Optional[float] = None
    field_y: Optional[float] = None
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    confidence: float = 0.0
    age_frames: int = 0


class CaptureSourceMetadata(BaseModel):
    kind: str = "unknown"
    session_id: Optional[str] = None
    stream_epoch: Optional[int] = None
    source_frame_id: Optional[int] = None
    t_us: Optional[int] = None
    transport_pts90k: Optional[int] = None
    camera_motion: Optional[dict[str, Any]] = None
    pose_missing_reason: Optional[str] = None
    backend_received_at_ms: Optional[float] = None


class GameEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: str
    confidence: float
    severity: str
    timestamp: float
    frame_id: int
    field_x: Optional[float] = None
    field_y: Optional[float] = None
    reviewed: bool = False
    foul_details: Optional[dict[str, Any]] = None
    involved_track_ids: list[int] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class FrameState(BaseModel):
    type: str = "frame_state"
    frame_id: int
    capture_timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    processed_timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    processing_fps: float = 0.0
    homography_status: HomographyStatus = HomographyStatus.UNAVAILABLE
    players: list[PlayerState] = Field(default_factory=list)
    ball: Optional[BallState] = None
    possession_track_id: Optional[int] = None
    events: list[GameEvent] = Field(default_factory=list)
    capture_source: Optional[CaptureSourceMetadata] = None


class MetricsSnapshot(BaseModel):
    type: str = "metrics"
    processing_fps: float = 0.0
    input_fps: float = 0.0
    inference_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    dropped_frames: int = 0
    queue_length: int = 0
    player_count: int = 0
    source_status: SourceStatus = SourceStatus.DISCONNECTED
    memory_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None
    player_inference_latency_ms: float = 0.0
    pitch_inference_latency_ms: float = 0.0
    pitch_detection_count: int = 0
    homography_reuse_ratio: float = 0.0
    homography_available_ratio: float = 0.0
    camera_motion_refresh_count: int = 0
    track_id_interruptions: int = 0
    track_occlusion_events: int = 0
    track_predicted_frames: int = 0
    track_recovered_count: int = 0
    track_reactivated_count: int = 0
    track_id_switches: int = 0
    track_fragmentations: int = 0
    track_max_missing_frames: int = 0
    track_entity_rebinds: int = 0
    track_entity_fragmentations: int = 0
    track_lifecycle_counts: dict[str, int] = Field(default_factory=dict)
    semantic_inference_count: int = 0
    semantic_label_switches: int = 0
    team_inference_count: int = 0
    team_unknown_rate: float = 0.0
    team_label_switches: int = 0
    jpeg_frames_encoded: int = 0
    jpeg_encode_latency_ms: float = 0.0
    foul_inference_count: int = 0
    decoded_frames: int = 0
    decode_dropped_frames: int = 0
    join_missing_frames: int = 0
    inference_dropped_frames: int = 0
    field_session_id: Optional[str] = None
    field_stream_epoch: Optional[int] = None
    decode_latency_ms: float = 0.0
    join_latency_ms: float = 0.0


class PipelineConfig(BaseModel):
    mode: str = "realtime"
    video_source: str = ""
    enable_foul_detection: bool = False
    enable_recording: bool = False
    target_video_path: str = ""
    show_keypoints: bool = True
    show_tracking_boxes: bool = True
    show_2d_projection: bool = True
    foul_confidence_threshold: float = 0.48
    device: str = "cpu"
    inference_backend: str = "auto"
    player_model_path: str = "assets/weights/yolo11s.pt"
    pitch_model_path: str = "assets/weights/football-pitch-detection.pt"
    role_model_path: str = "assets/weights/player-role-yolo11n.pt"
    team_classifier_path: Optional[str] = None
    team_calibration_path: Optional[str] = None
    require_team_calibration: bool = True
    role_detection_interval: int = Field(3, ge=1)
    team_classification_interval: int = Field(5, ge=1)
    track_activation_threshold: float = Field(0.25, ge=0.0, le=1.0)
    track_lost_buffer: int = Field(45, ge=1)
    track_matching_threshold: float = Field(0.8, ge=0.0, le=1.0)
    track_minimum_consecutive_frames: int = Field(2, ge=1)
    camera_calibration_path: str = "assets/calibration/camera.npz"
    enable_undistortion: bool = True
    calibration_alpha: float = Field(0.0, ge=0.0, le=1.0)
    pitch_detection_interval: int = Field(5, ge=1)
    imgsz: int = Field(640, ge=32)
    player_confidence: float = Field(0.25, ge=0.0, le=1.0)
    player_iou: float = Field(0.7, ge=0.0, le=1.0)
    max_prediction_gap_frames: int = Field(6, ge=0)
    track_reactivation_window_frames: int = Field(12, ge=0)


class PipelineCommand(BaseModel):
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
