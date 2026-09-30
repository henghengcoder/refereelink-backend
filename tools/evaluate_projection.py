"""Evaluate 2D player projection stability and accuracy proxies on local videos.

``collect`` runs the shared inference pipeline (same settings as
``render_diagnostic_video.py``) and writes, per video, into ``debug/diag/<name>/``:
the diagnostic MP4, a contact sheet, a JSON metrics report and a per-frame
``<name>_frames.jsonl`` dump of projected player positions and homography fit
quality. ``analyze`` turns those dumps into stability/accuracy-proxy metrics in
``debug/diag/summary/projection_metrics.json``.

Usage::

    uv run python tools/evaluate_projection.py collect 0bfacc 2e57b9
    uv run python tools/evaluate_projection.py analyze
    uv run python tools/evaluate_projection.py --out-dir debug/diag2 collect

Without video names both commands process every ``assets/data/*_0.mp4``.
``--out-dir`` (default ``debug/diag``) keeps each iteration's results apart.
There is no ground truth, so accuracy is measured through proxies: physically
implausible speeds and the jump between consecutive homographies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "assets" / "data"
DIAG_DIR = REPO_ROOT / "debug" / "diag"

FPS = 25.0
CM_PER_M = 100.0  # SoccerPitchConfiguration uses centimetres
SPEED_LIMIT_MPS = 10.0  # above a human sprint
SMOOTH_WINDOW = 5


def _default_videos() -> list[str]:
    return sorted(path.name[: -len("_0.mp4")] for path in DATA_DIR.glob("*_0.mp4"))


def _load_render_module():
    spec = importlib.util.spec_from_file_location(
        "render_diagnostic_video", REPO_ROOT / "tools" / "render_diagnostic_video.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(name: str, device: str, diag_dir: Path) -> None:
    import cv2
    import torch

    from app.constants.paths import (
        CAMERA_CALIBRATION_PATH,
        PITCH_DETECTION_MODEL_PATH,
        PLAYER_DETECTION_MODEL_PATH,
    )
    from app.pipeline.engine import InferencePipeline
    from app.pipeline.source import LocalFileSource
    from app.state.store import StateStore
    from app.vision import core as vision_core

    render = _load_render_module()
    input_path = DATA_DIR / f"{name}_0.mp4"
    out_dir = diag_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wrap VisionCore at runtime to capture, per frame, both the raw keypoint
    # fit and the (possibly stabilised) homography actually used.
    projection_log: list[dict] = []
    raw_fit: dict = {}
    original_process = vision_core.VisionCore.process
    original_projection = vision_core.VisionCore._projection_for_frame

    def projection_and_record(self, frame, frame_index, force_refresh=False):
        result = original_projection(self, frame, frame_index, force_refresh=force_refresh)
        raw_fit["H"] = (
            None
            if result.homography is None
            else np.asarray(result.homography, dtype=float).tolist()
        )
        raw_fit["forced"] = bool(force_refresh)
        return result

    def process_and_record(self, frame, frame_index):
        raw_fit.clear()
        vision_frame = original_process(self, frame, frame_index)
        projection = vision_frame.projection
        projection_log.append(
            {
                "frame_index": int(frame_index),
                "status": projection.homography_status,
                "reproj_px": (
                    None
                    if projection.reprojection_error is None
                    else float(projection.reprojection_error)
                ),
                "n_keypoints": len(projection.tracking_observations),
                "H": (
                    None
                    if projection.homography is None
                    else np.asarray(projection.homography, dtype=float).tolist()
                ),
                "raw_H": raw_fit.get("H"),
                "forced": raw_fit.get("forced", False),
            }
        )
        return vision_frame

    vision_core.VisionCore.process = process_and_record
    vision_core.VisionCore._projection_for_frame = projection_and_record

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or FPS
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    store = StateStore()
    video_path = out_dir / f"{name}.mp4"
    writer = render.DiagnosticVideoWriter(video_path, fps, width, height)
    sequence = {"index": 0}

    with open(out_dir / f"{name}_frames.jsonl", "w", encoding="utf-8") as frames_file:

        def frame_sink(frame, state):
            writer(frame, state)
            index = sequence["index"]
            sequence["index"] += 1
            record = {
                "frame_id": state.frame_id,
                "homography_status": state.homography_status.value,
                "proj": projection_log[index] if index < len(projection_log) else {},
                "players": [
                    {
                        "track_id": player.track_id,
                        "entity_id": player.entity_id,
                        "x": player.field_x,
                        "y": player.field_y,
                        "bbox": player.bbox,
                        "conf": player.confidence,
                        "track_status": player.track_status,
                    }
                    for player in state.players
                ],
            }
            frames_file.write(json.dumps(record) + "\n")

        pipeline = InferencePipeline(
            source=LocalFileSource(str(input_path), store=store),
            store=store,
            device=device,
            player_model_path=PLAYER_DETECTION_MODEL_PATH,
            pitch_model_path=PITCH_DETECTION_MODEL_PATH,
            camera_calibration_path=CAMERA_CALIBRATION_PATH,
            enable_undistortion=True,
            pitch_detection_interval=5,
            imgsz=640,
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
            writer.close()
            vision_core.VisionCore.process = original_process
            vision_core.VisionCore._projection_for_frame = original_projection
        elapsed = time.monotonic() - started

    render._make_contact_sheet(video_path, out_dir / f"{name}.jpg")
    report = {
        "input": str(input_path),
        "source_frames": source_frames,
        "rendered_frames": writer.frames_written,
        "projection_calls": len(projection_log),
        "source_fps": fps,
        "elapsed_sec": round(elapsed, 3),
        "render_fps": round(writer.frames_written / max(elapsed, 0.001), 2),
        "metrics": store.metrics.model_dump(mode="json"),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (out_dir / f"{name}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"DONE {name}: {writer.frames_written}/{source_frames} frames, "
        f"{len(projection_log)} projections, {elapsed:.1f}s"
    )


def _percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def _bottom_center(bbox):
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def _apply_homography(homography, points):
    points = np.asarray(points, dtype=float)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ np.asarray(homography).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _split_consecutive(frames: list[int]) -> list[list[int]]:
    segments, current = [], [frames[0]]
    for frame in frames[1:]:
        if frame == current[-1] + 1:
            current.append(frame)
        else:
            segments.append(current)
            current = [frame]
    segments.append(current)
    return segments


def analyze_video(name: str, diag_dir: Path) -> dict:
    out_dir = diag_dir / name
    with open(out_dir / f"{name}_frames.jsonl", encoding="utf-8") as frames_file:
        rows = [json.loads(line) for line in frames_file]
    report = json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    frame_count = len(rows)

    status = Counter(row["homography_status"] for row in rows)
    total_players = sum(len(row["players"]) for row in rows)
    projected_per_frame = np.array(
        [sum(player["x"] is not None for player in row["players"]) for row in rows]
    )

    fresh = [row["proj"] for row in rows if row["proj"].get("status") == "fresh"]
    reprojection = np.array([f["reproj_px"] for f in fresh if f["reproj_px"] is not None])
    keypoints = np.array([f["n_keypoints"] for f in fresh])

    # Trajectories keyed by track_id -> {frame_id: (x, y)} in metres.
    trajectories: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        for player in row["players"]:
            if player["x"] is not None:
                trajectories[player["track_id"]][row["frame_id"]] = (
                    player["x"] / CM_PER_M,
                    player["y"] / CM_PER_M,
                )

    speeds, accelerations, residuals, per_track_jitter, track_lengths = [], [], [], [], []
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    half = SMOOTH_WINDOW // 2
    for points in trajectories.values():
        frames = sorted(points)
        track_lengths.append(len(frames))
        track_residuals = []
        for segment in _split_consecutive(frames):
            xy = np.array([points[frame] for frame in segment])
            if len(xy) >= 2:
                speeds.extend(np.linalg.norm(np.diff(xy, axis=0), axis=1) * FPS)
            if len(xy) >= 3:
                accelerations.extend(
                    np.linalg.norm(np.diff(xy, 2, axis=0), axis=1) * FPS * FPS
                )
            if len(xy) >= SMOOTH_WINDOW:
                smoothed = np.column_stack(
                    [np.convolve(xy[:, axis], kernel, mode="valid") for axis in range(2)]
                )
                residual = np.linalg.norm(xy[half:-half] - smoothed, axis=1)
                residuals.extend(residual)
                track_residuals.extend(residual)
        if len(track_residuals) >= 10:
            per_track_jitter.append(float(np.sqrt(np.mean(np.square(track_residuals)))))
    speeds = np.array(speeds)
    accelerations = np.array(accelerations)
    residuals = np.array(residuals)

    # Homography refresh jump: the same foot point under the previous vs the new H.
    jumps, jumps_after_reuse = [], []
    for previous, current in zip(rows, rows[1:]):
        if (
            current["proj"].get("status") != "fresh"
            or current["proj"].get("H") is None
            or previous["proj"].get("H") is None
        ):
            continue
        feet = [
            _bottom_center(player["bbox"])
            for player in current["players"]
            if player["x"] is not None and player["bbox"]
        ]
        if not feet:
            continue
        distance = (
            np.linalg.norm(
                _apply_homography(current["proj"]["H"], feet)
                - _apply_homography(previous["proj"]["H"], feet),
                axis=1,
            )
            / CM_PER_M
        )
        jumps.extend(distance)
        if previous["proj"].get("status") == "reused":
            jumps_after_reuse.append(float(np.median(distance)))
    jumps = np.array(jumps)

    # Drift proxy: on fresh frames, how far the homography actually used is
    # from the raw keypoint fit of that frame (0 when nothing is stabilised).
    fit_gaps = []
    for row in rows:
        proj = row["proj"]
        if proj.get("status") != "fresh" or proj.get("H") is None or not proj.get("raw_H"):
            continue
        feet = [
            _bottom_center(player["bbox"])
            for player in row["players"]
            if player["x"] is not None and player["bbox"]
        ]
        if feet:
            gap = np.linalg.norm(
                _apply_homography(proj["H"], feet) - _apply_homography(proj["raw_H"], feet),
                axis=1,
            )
            fit_gaps.append(float(np.median(gap)) / CM_PER_M)
    fit_gaps = np.array(fit_gaps)

    return {
        "video": name,
        "frames": frame_count,
        "render_fps": report["render_fps"],
        "latency_ms": metrics["end_to_end_latency_ms"],
        "player_ms": metrics["player_inference_latency_ms"],
        "pitch_ms": metrics["pitch_inference_latency_ms"],
        "gpu_mem_mb": metrics["gpu_memory_mb"],
        "h_fresh": status["fresh"] / frame_count,
        "h_reused": status["reused"] / frame_count,
        "h_stale": status["stale"] / frame_count,
        "h_unavail": status["unavailable"] / frame_count,
        "motion_refresh": metrics["camera_motion_refresh_count"],
        "players_per_frame": total_players / frame_count,
        "proj_rate": float(projected_per_frame.sum()) / max(total_players, 1),
        "proj_per_frame_mean": float(projected_per_frame.mean()),
        "proj_per_frame_std": float(projected_per_frame.std()),
        "reproj_mean": float(reprojection.mean()) if len(reprojection) else None,
        "reproj_p95": _percentile(reprojection, 95),
        "kp_mean": float(keypoints.mean()) if len(keypoints) else None,
        "kp_min4_rate": float(np.mean(keypoints <= 4)) if len(keypoints) else None,
        "speed_p50": _percentile(speeds, 50),
        "speed_p95": _percentile(speeds, 95),
        "speed_over_limit": float(np.mean(speeds > SPEED_LIMIT_MPS)) if len(speeds) else None,
        "jitter_rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "jitter_p95": _percentile(residuals, 95),
        "jitter_track_median": float(np.median(per_track_jitter)) if per_track_jitter else None,
        "accel_p50": _percentile(accelerations, 50),
        "accel_p95": _percentile(accelerations, 95),
        "hjump_mean": float(jumps.mean()) if len(jumps) else None,
        "hjump_p95": _percentile(jumps, 95),
        "hjump_after_reuse_median": (
            float(np.median(jumps_after_reuse)) if jumps_after_reuse else None
        ),
        "hjump_over_1m": float(np.mean(jumps > 1.0)) if len(jumps) else None,
        "fit_gap_median": float(np.median(fit_gaps)) if len(fit_gaps) else None,
        "fit_gap_p95": _percentile(fit_gaps, 95),
        "tracks": len(trajectories),
        "track_len_median": float(np.median(track_lengths)),
        "track_len_mean": float(np.mean(track_lengths)),
        "id_switches": metrics["track_id_switches"],
        "interruptions": metrics["track_id_interruptions"],
        "fragmentations": metrics["track_fragmentations"],
        "team_unknown_rate": metrics["team_unknown_rate"],
    }


def analyze(names: list[str], diag_dir: Path) -> None:
    results = [analyze_video(name, diag_dir) for name in names]
    summary_dir = diag_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "projection_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("metric".ljust(26) + "".join(result["video"].rjust(10) for result in results))
    for key in list(results[0])[1:]:
        cells = []
        for result in results:
            value = result[key]
            if value is None:
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.3f}"
            else:
                text = str(value)
            cells.append(text.rjust(10))
        print(key.ljust(26) + "".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DIAG_DIR,
        help="results directory, relative to the repo root (default: debug/diag)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect", help="run the pipeline and dump frames")
    collect_parser.add_argument("videos", nargs="*", help="names like 0bfacc (default: all)")
    collect_parser.add_argument("--device", default="cuda")
    analyze_parser = subparsers.add_parser("analyze", help="compute metrics from dumps")
    analyze_parser.add_argument("videos", nargs="*", help="names like 0bfacc (default: all)")
    args = parser.parse_args()

    diag_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    names = args.videos or _default_videos()
    if args.command == "collect":
        for name in names:
            collect(name, args.device, diag_dir)
    else:
        analyze(names, diag_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
