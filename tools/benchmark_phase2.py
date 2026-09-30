#!/usr/bin/env python3
"""Reproducible CPU benchmark for phase-two entity components.

This benchmark intentionally uses fake detections and synthetic crops. It
measures the scheduling and state-management overhead without requiring model
weights, so it is useful in CI and before hardware is fixed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import supervision as sv

from app.classification.online import OnlineTeamClassifier
from app.vision.semantics import TrackSemanticManager


def _crop(color: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((48, 24, 3), dtype=np.uint8)
    result[:] = color
    return result


def run(frames: int = 300, semantic_interval: int = 5) -> dict:
    red = _crop((0, 0, 220))
    blue = _crop((220, 0, 0))
    team_classifier = OnlineTeamClassifier(warmup_frames=2, warmup_stride=1)
    team_classifier.fit([red, blue], labels=[0, 1])
    semantics = TrackSemanticManager(team_classifier=team_classifier)

    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 30, 70], [60, 10, 80, 70]], dtype=np.float32),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        tracker_id=np.array([1, 2]),
    )
    start = time.perf_counter()
    semantic_calls = 0
    for frame_index in range(1, frames + 1):
        if frame_index == 1 or frame_index % semantic_interval == 0:
            semantics.update(frame, detections, frame_index)
            semantic_calls += 1
    elapsed = time.perf_counter() - start
    return {
        "frames": frames,
        "elapsed_ms": round(elapsed * 1000.0, 3),
        "component_fps": round(frames / max(elapsed, 1e-9), 2),
        "semantic_interval": semantic_interval,
        "semantic_inference_count": semantic_calls,
        "semantic_label_switches": semantics.semantic_label_switches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--semantic-interval", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.frames, args.semantic_interval)
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
