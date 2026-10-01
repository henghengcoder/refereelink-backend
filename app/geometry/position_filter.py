"""Per-player temporal filtering of projected pitch coordinates.

Bottom-centre anchors of small, distant bounding boxes jitter by a few pixels
every frame, which the homography amplifies into metre-scale jumps on the 2D
pitch.  ``PlayerPositionFilter`` runs one constant-velocity Kalman filter per
player entity in pitch coordinates (centimetres), rejects physically
implausible single-frame jumps, and never invents positions for players that
have no measurement in the current frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class _TrackState:
    state: np.ndarray  # [x, y, vx, vy] in cm and cm/s
    covariance: np.ndarray  # 4x4
    last_frame: int
    outlier_streak: int = 0


class PlayerPositionFilter:
    """Constant-velocity Kalman filter per player with outlier gating.

    Measurement noise scales inversely with the bounding-box height, so far
    players (small boxes, larger pixel-to-metre ratio) are trusted less.
    """

    def __init__(
        self,
        fps: float = 25.0,
        *,
        acceleration_std_cm_s2: float = 300.0,
        measurement_std_cm: float = 50.0,
        reference_box_height_px: float = 100.0,
        min_measurement_std_cm: float = 15.0,
        max_measurement_std_cm: float = 150.0,
        gate_min_cm: float = 120.0,
        gate_max_cm: float = 250.0,
        gate_sigma: float = 4.0,
        max_outlier_streak: int = 3,
        max_speed_cm_s: float = 1000.0,
        max_missing_frames: int = 12,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.acceleration_std = float(acceleration_std_cm_s2)
        self.measurement_std = float(measurement_std_cm)
        self.reference_box_height = float(reference_box_height_px)
        self.min_measurement_std = float(min_measurement_std_cm)
        self.max_measurement_std = float(max_measurement_std_cm)
        self.gate_min = float(gate_min_cm)
        self.gate_max = max(float(gate_max_cm), self.gate_min)
        self.gate_sigma = float(gate_sigma)
        self.max_outlier_streak = max(int(max_outlier_streak), 1)
        self.max_speed = float(max_speed_cm_s)
        self.max_missing_frames = max(int(max_missing_frames), 1)
        self._tracks: dict[int, _TrackState] = {}
        self.outliers_rejected = 0
        self.resets = 0

    @property
    def active_keys(self) -> set[int]:
        return set(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        keys: Sequence[int],
        field_xy: np.ndarray,
        frame_index: int,
        box_heights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Filter one frame of measurements.

        ``field_xy`` rows that are NaN (no projection) stay NaN in the output
        and do not advance that player's filter.
        """

        measurements = np.asarray(field_xy, dtype=np.float64).reshape(-1, 2)
        output = np.full(measurements.shape, np.nan, dtype=np.float64)
        self._expire(frame_index)
        for index, key in enumerate(keys):
            measurement = measurements[index]
            if not np.isfinite(measurement).all():
                continue
            height = None
            if box_heights is not None:
                height = float(box_heights[index])
            output[index] = self._update_one(int(key), measurement, frame_index, height)
        return output.astype(np.asarray(field_xy).dtype, copy=False)

    def _measurement_variance(self, box_height: Optional[float]) -> float:
        std = self.measurement_std
        if box_height is not None and box_height > 0:
            std = self.measurement_std * self.reference_box_height / box_height
        std = min(max(std, self.min_measurement_std), self.max_measurement_std)
        return std * std

    def _new_track(self, measurement: np.ndarray, frame_index: int, variance: float) -> _TrackState:
        velocity_variance = self.max_speed * self.max_speed
        return _TrackState(
            state=np.array([measurement[0], measurement[1], 0.0, 0.0]),
            covariance=np.diag([variance, variance, velocity_variance, velocity_variance]),
            last_frame=frame_index,
        )

    def _update_one(
        self,
        key: int,
        measurement: np.ndarray,
        frame_index: int,
        box_height: Optional[float],
    ) -> np.ndarray:
        variance = self._measurement_variance(box_height)
        track = self._tracks.get(key)
        if track is None or frame_index <= track.last_frame:
            track = self._new_track(measurement, frame_index, variance)
            self._tracks[key] = track
            return track.state[:2].copy()

        dt = (frame_index - track.last_frame) / self.fps
        transition = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        q = self.acceleration_std**2
        block = np.array([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]]) * q
        process = np.zeros((4, 4))
        process[np.ix_([0, 2], [0, 2])] = block
        process[np.ix_([1, 3], [1, 3])] = block

        predicted_state = transition @ track.state
        predicted_cov = transition @ track.covariance @ transition.T + process

        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        innovation = measurement - observation @ predicted_state
        innovation_cov = observation @ predicted_cov @ observation.T + np.eye(2) * variance
        # Noise-scaled gate, bounded: even for tiny distant boxes a multi-metre
        # jump within one frame is physically impossible.
        sigma = float(np.sqrt(np.trace(innovation_cov) / 2.0))
        gate = min(max(self.gate_min, self.gate_sigma * sigma), self.gate_max)

        track.last_frame = frame_index
        if float(np.linalg.norm(innovation)) > gate:
            track.outlier_streak += 1
            self.outliers_rejected += 1
            if track.outlier_streak >= self.max_outlier_streak:
                # A persistent jump is real (ID rebind, homography correction).
                self.resets += 1
                track = self._new_track(measurement, frame_index, variance)
                self._tracks[key] = track
                return track.state[:2].copy()
            track.state = predicted_state
            track.covariance = predicted_cov
            return predicted_state[:2].copy()

        track.outlier_streak = 0
        gain = predicted_cov @ observation.T @ np.linalg.inv(innovation_cov)
        track.state = predicted_state + gain @ innovation
        track.covariance = (np.eye(4) - gain @ observation) @ predicted_cov
        speed = float(np.hypot(track.state[2], track.state[3]))
        if speed > self.max_speed:
            track.state[2:] *= self.max_speed / speed
        return track.state[:2].copy()

    def _expire(self, frame_index: int) -> None:
        stale = [
            key
            for key, track in self._tracks.items()
            if frame_index - track.last_frame > self.max_missing_frames
        ]
        for key in stale:
            del self._tracks[key]
