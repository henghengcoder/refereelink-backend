from __future__ import annotations

import time

import numpy as np
import supervision as sv

from app.geometry.pitch_projection import PitchProjectionResult
from app.field_ingest.frames import CapturedFrame
from app.pipeline.engine import InferencePipeline
from app.pipeline.recorder import VideoRecorder
from app.state.models import PlayerRole, TeamLabel
from app.state.store import StateStore
from app.vision.core import VisionFrame
from app.vision.semantics import SemanticResult


class _Source:
    frame_count = 1


class _ProjectionEngine:
    class _Config:
        length = 12000.0
        width = 7000.0

    config = _Config()


class _VisionCore:
    projection_engine = _ProjectionEngine()
    frames_processed = 1


class _SemanticManager:
    semantic_label_switches = 0

    def update(self, *, frame, detections, frame_index):
        del frame, detections, frame_index
        return {
            7: SemanticResult(
                track_id=7,
                role="outfield",
                team=TeamLabel.HOME,
                team_id=0,
                role_confidence=0.91,
                team_confidence=0.88,
                status="stable",
            )
        }


def test_pipeline_maps_semantics_and_field_coordinates_to_frame_state(tmp_path) -> None:
    detections = sv.Detections(
        xyxy=np.array([[8, 8, 12, 12]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        tracker_id=np.array([7]),
    )
    projection = PitchProjectionResult(
        tracking_observations=[],
        projected_keypoints=[],
        homography=np.eye(3, dtype=np.float32),
        homography_status="fresh",
        reprojection_error=0.0,
    )
    vision_frame = VisionFrame(
        undistorted_frame=np.zeros((100, 100, 3), dtype=np.uint8),
        detections=detections,
        tracked_detections=detections,
        projection=projection,
        field_xy=np.array([[10.0, 12.0]], dtype=np.float32),
        color_lookup=np.array([4], dtype=np.int64),
        person_only=True,
    )

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline._source = _Source()
    pipeline._store = StateStore()
    pipeline._vision_core = _VisionCore()
    pipeline._semantic_manager = _SemanticManager()
    pipeline._semantic_interval = 1
    pipeline._semantic_last_frame = None
    pipeline._semantic_results = {}
    pipeline.semantic_inference_count = 0
    pipeline._metrics_start = time.monotonic()
    pipeline._metrics_frames = 0
    pipeline._recorder = VideoRecorder(str(tmp_path / "annotated.mp4"), pipeline._store, fps=10.0)
    pipeline._recorder.start()

    pipeline._vision_core.process = lambda frame, frame_index: vision_frame

    captured = CapturedFrame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        session_id="session",
        stream_epoch=3,
        source_frame_id=42,
        t_us=1_200_000,
        transport_pts90k=108_000,
        capture_unix_us=1_000_000,
        camera_motion={"pitch": 0.2},
        pose_missing_reason=None,
        backend_received_at="2026-09-20T00:00:00Z",
    )
    frame_state = pipeline._process_frame(captured.image, 1000.0, captured)

    assert frame_state is not None
    assert frame_state.players[0].role == PlayerRole.PLAYER
    assert frame_state.players[0].team == TeamLabel.HOME
    assert frame_state.players[0].team_id == 0
    assert frame_state.players[0].bbox == (8.0, 8.0, 12.0, 12.0)
    assert frame_state.players[0].semantic_status == "stable"
    assert frame_state.players[0].field_x == 10.0
    assert frame_state.players[0].field_y == 12.0
    assert not hasattr(frame_state, "ball")
    assert frame_state.capture_source is not None
    assert frame_state.capture_source.stream_epoch == 3
    assert frame_state.capture_source.source_frame_id == 42
    assert frame_state.capture_source.transport_pts90k == 108_000
    pipeline._recorder.stop()
    assert pipeline._recorder.frames_written == 1
    assert (tmp_path / "annotated.mp4").is_file()


def test_pipeline_rebinds_semantic_history_before_new_track_update() -> None:
    class RebindingSemanticManager:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def rebind_track(self, new_track_id: int, old_track_id: int) -> bool:
            self.calls.append((new_track_id, old_track_id))
            return True

        def update(self, *, frame, detections, frame_index):
            del frame, detections, frame_index
            return {
                99: SemanticResult(
                    track_id=99,
                    role="outfield",
                    team=TeamLabel.AWAY,
                    team_id=1,
                    role_confidence=0.9,
                    team_confidence=0.9,
                    status="stable",
                )
            }

    pipeline = InferencePipeline.__new__(InferencePipeline)
    manager = RebindingSemanticManager()
    pipeline._semantic_manager = manager
    pipeline._semantic_interval = 1
    pipeline._semantic_last_frame = None
    pipeline._semantic_results = {}
    pipeline.semantic_inference_count = 0
    detections = sv.Detections(
        xyxy=np.array([[8, 8, 12, 12]], dtype=np.float32),
        tracker_id=np.array([99]),
    )

    results = pipeline._update_semantics(
        frame=np.zeros((20, 20, 3), dtype=np.uint8),
        detections=detections,
        frame_index=4,
        rebindings={99: 7},
    )

    assert manager.calls == [(99, 7)]
    assert results[99].team == TeamLabel.AWAY
    assert pipeline.semantic_inference_count == 1
