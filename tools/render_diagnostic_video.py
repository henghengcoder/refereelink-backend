"""Render a diagnostic video from the shared inference pipeline.

The output keeps the annotated camera view and adds a right-hand panel with
the projected pitch coordinates and homography status.  It is intended for
debugging a real input video, not as a low-latency production recorder.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.annotators.pitch import draw_pitch
from app.config.pitch import SoccerPitchConfiguration
from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
)
from app.pipeline.engine import InferencePipeline
from app.pipeline.source import LocalFileSource
from app.state.models import FrameState
from app.state.store import StateStore


LOGGER = logging.getLogger("render_diagnostic_video")
PITCH_CONFIG = SoccerPitchConfiguration()


def _color_for_team(team_id: int) -> tuple[int, int, int]:
    return {0: (147, 20, 255), 1: (255, 191, 0)}.get(team_id, (160, 160, 160))


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_panel(frame: np.ndarray, state: FrameState, panel_width: int) -> np.ndarray:
    """Append a compact pitch/radar and status panel to an annotated frame."""

    height = frame.shape[0]
    panel = np.full((height, panel_width, 3), (24, 27, 31), dtype=np.uint8)
    cv2.line(panel, (0, 0), (0, height - 1), (90, 90, 90), 2)

    status = state.homography_status.value.upper()
    status_color = {
        "FRESH": (80, 220, 120),
        "REUSED": (0, 215, 255),
        "STALE": (0, 140, 255),
        "UNAVAILABLE": (80, 80, 230),
    }.get(status, (180, 180, 180))
    _draw_text(panel, "VISION DIAGNOSTICS", (18, 28), (245, 245, 245), 0.65, 2)
    _draw_text(panel, f"frame: {state.frame_id}", (18, 58))
    _draw_text(panel, f"homography: {status}", (18, 82), status_color, 0.55, 2)

    valid_players = sum(
        player.field_x is not None and player.field_y is not None for player in state.players
    )
    ball_status = state.ball.status.value if state.ball is not None else "unavailable"
    _draw_text(
        panel,
        f"players: {len(state.players)}  projected: {valid_players}",
        (18, 108),
    )
    _draw_text(
        panel,
        f"ball: {ball_status}",
        (18, 132),
        (0, 215, 255) if ball_status != "unavailable" else (160, 160, 160),
    )

    padding = 18
    pitch_top = 160
    pitch_bottom = height - 28
    scale = min(
        (panel_width - 2 * padding - 2) / PITCH_CONFIG.length,
        (pitch_bottom - pitch_top - 2 * padding) / PITCH_CONFIG.width,
    )
    scale = max(scale, 0.01)
    pitch = draw_pitch(
        PITCH_CONFIG,
        padding=padding,
        scale=scale,
        line_thickness=max(1, int(round(scale * 35))),
        point_radius=max(2, int(round(scale * 30))),
    )
    pitch_height, pitch_width = pitch.shape[:2]
    pitch_x = max((panel_width - pitch_width) // 2, 0)
    pitch_y = pitch_top + max((pitch_bottom - pitch_top - pitch_height) // 2, 0)
    y_end = min(pitch_y + pitch_height, height)
    x_end = min(pitch_x + pitch_width, panel_width)
    panel[pitch_y:y_end, pitch_x:x_end] = pitch[: y_end - pitch_y, : x_end - pitch_x]

    def project_to_panel(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(pitch_x + padding + x * scale)),
            int(round(pitch_y + padding + y * scale)),
        )

    for player in state.players:
        if player.field_x is None or player.field_y is None:
            continue
        point = project_to_panel(player.field_x, player.field_y)
        color = _color_for_team(player.team_id)
        cv2.circle(panel, point, 8, color, -1, cv2.LINE_AA)
        cv2.circle(panel, point, 9, (20, 20, 20), 1, cv2.LINE_AA)
        label = f"#{player.track_id} {player.field_x / 100:.1f},{player.field_y / 100:.1f}m"
        label_x = min(max(point[0] + 10, 4), panel_width - 170)
        label_y = min(max(point[1] - 8, 155), height - 8)
        _draw_text(panel, label, (label_x, label_y), color, 0.38)

    if state.ball is not None and state.ball.field_x is not None and state.ball.field_y is not None:
        point = project_to_panel(state.ball.field_x, state.ball.field_y)
        cv2.circle(panel, point, 8, (0, 215, 255), 2, cv2.LINE_AA)
        _draw_text(panel, "BALL", (point[0] + 10, point[1] + 5), (0, 215, 255), 0.4, 2)

    return np.hstack((frame, panel))


class DiagnosticVideoWriter:
    def __init__(self, output_path: Path, fps: float, width: int, height: int) -> None:
        self.output_path = output_path
        self.fps = fps if fps > 0 else 25.0
        self.panel_width = max(420, min(560, width // 3))
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width + self.panel_width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
        self.frames_written = 0

    def __call__(self, frame: np.ndarray, state: FrameState) -> None:
        combined = _render_panel(frame, state, self.panel_width)
        self.writer.write(combined)
        self.frames_written += 1

    def close(self) -> None:
        self.writer.release()


def _make_contact_sheet(
    video_path: Path,
    output_path: Path,
    count: int = 12,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return
    indices = np.linspace(0, total - 1, min(count, total), dtype=int)
    samples: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        thumb_width = 480
        thumb_height = max(1, int(frame.shape[0] * thumb_width / frame.shape[1]))
        thumbnail = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        _draw_text(thumbnail, f"frame {index + 1}", (12, 26), (255, 255, 255), 0.65, 2)
        samples.append(thumbnail)
    capture.release()
    if not samples:
        return

    columns = 3
    rows = (len(samples) + columns - 1) // columns
    cell_h = max(image.shape[0] for image in samples)
    cell_w = max(image.shape[1] for image in samples)
    sheet = np.full((rows * cell_h, columns * cell_w, 3), (18, 18, 18), dtype=np.uint8)
    for position, image in enumerate(samples):
        row, column = divmod(position, columns)
        y = row * cell_h
        x = column * cell_w
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    cv2.imwrite(str(output_path), sheet)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output diagnostic MP4 path")
    parser.add_argument("--contact-sheet", required=True, help="Output JPG contact sheet path")
    parser.add_argument("--report", required=True, help="Output JSON report path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--player-model-path", default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-model-path", default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--camera-calibration-path", default=CAMERA_CALIBRATION_PATH)
    parser.add_argument("--pitch-detection-interval", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--disable-undistortion", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    contact_sheet_path = Path(args.contact_sheet).resolve()
    report_path = Path(args.report).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_capture = cv2.VideoCapture(str(input_path))
    if not metadata_capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")
    width = int(metadata_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(metadata_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(metadata_capture.get(cv2.CAP_PROP_FPS)) or 25.0
    source_frames = int(metadata_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata_capture.release()

    store = StateStore()
    source = LocalFileSource(str(input_path), store=store)
    sink = DiagnosticVideoWriter(output_path, fps, width, height)
    pipeline: Optional[InferencePipeline] = None

    def frame_sink(frame: np.ndarray, state: FrameState) -> None:
        sink(frame, state)

    pipeline = InferencePipeline(
        source=source,
        store=store,
        device=args.device,
        player_model_path=args.player_model_path,
        pitch_model_path=args.pitch_model_path,
        camera_calibration_path=args.camera_calibration_path,
        enable_undistortion=not args.disable_undistortion,
        pitch_detection_interval=args.pitch_detection_interval,
        imgsz=args.imgsz,
        enable_foul_detection=False,
        frame_sink=frame_sink,
    )

    started = time.monotonic()
    try:
        pipeline.start()
        if pipeline._thread is not None:  # The local-file worker terminates at EOF.
            pipeline._thread.join()
    finally:
        pipeline.stop()
        sink.close()

    elapsed = time.monotonic() - started
    _make_contact_sheet(output_path, contact_sheet_path)
    metrics = store.metrics.model_dump(mode="json")
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "contact_sheet": str(contact_sheet_path),
        "source_frames": source_frames,
        "rendered_frames": sink.frames_written,
        "source_fps": fps,
        "elapsed_sec": round(elapsed, 3),
        "render_fps": round(sink.frames_written / max(elapsed, 0.001), 2),
        "metrics": metrics,
        "cuda_available": bool(__import__("torch").cuda.is_available()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Rendered %d/%d frames to %s", sink.frames_written, source_frames, output_path)
    LOGGER.info("Contact sheet: %s", contact_sheet_path)
    LOGGER.info("Report: %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
