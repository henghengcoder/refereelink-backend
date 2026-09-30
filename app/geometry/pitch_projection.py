from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import supervision as sv

from app.config.pitch import SoccerPitchConfiguration


MIN_KEYPOINT_CONFIDENCE = 0.35
MIN_KEYPOINTS_FOR_HOMOGRAPHY = 4
MIN_INLIERS_FOR_HOMOGRAPHY = 4
MAX_REPROJECTION_ERROR_PX = 12.0
FRAME_EDGE_MARGIN_PX = 8
MAX_STALE_SECONDS = 0.5


@dataclass(frozen=True)
class PitchPointReference:
    index: int
    label: str
    world_xy: Tuple[float, float]
    color: str


@dataclass(frozen=True)
class PitchKeypointObservation:
    reference: PitchPointReference
    image_xy: Tuple[float, float]
    confidence: float
    source: str


@dataclass(frozen=True)
class ProjectedPitchKeypoint:
    reference: PitchPointReference
    projected_world_xy: Tuple[float, float]
    source: str


@dataclass
class PitchProjectionResult:
    tracking_observations: List[PitchKeypointObservation]
    projected_keypoints: List[ProjectedPitchKeypoint]
    homography: Optional[np.ndarray]
    homography_status: str
    reprojection_error: Optional[float]

    @property
    def available(self) -> bool:
        return self.homography is not None


def build_pitch_point_references(
    config: SoccerPitchConfiguration,
) -> List[PitchPointReference]:
    return [
        PitchPointReference(
            index=index,
            label=config.labels[index] if index < len(config.labels) else str(index + 1),
            world_xy=(
                float(config.vertices[index][0]),
                float(config.vertices[index][1]),
            ),
            color=config.colors[index % len(config.colors)],
        )
        for index in range(len(config.vertices))
    ]


def translate_homography(
    homography: np.ndarray,
    shift_xy: Tuple[float, float],
) -> np.ndarray:
    """Compensate a pure image translation since ``homography`` was fitted.

    ``shift_xy`` is the displacement of image content in the current frame
    relative to the reference frame (``CameraMotionEstimator`` convention), so a
    current pixel ``p`` corresponds to ``p - shift`` in the reference frame.
    """

    translation = np.array(
        [[1.0, 0.0, -float(shift_xy[0])], [0.0, 1.0, -float(shift_xy[1])], [0.0, 0.0, 1.0]]
    )
    return np.asarray(homography, dtype=np.float64) @ translation


def homography_deviation(
    previous: np.ndarray,
    current: np.ndarray,
    image_points: np.ndarray,
) -> Optional[float]:
    """Median pitch-space distance between two homographies at image points."""

    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return None
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    projected = []
    for matrix in (previous, current):
        mapped = homogeneous @ np.asarray(matrix, dtype=np.float64).T
        with np.errstate(divide="ignore", invalid="ignore"):
            projected.append(mapped[:, :2] / mapped[:, 2:3])
    distances = np.linalg.norm(projected[1] - projected[0], axis=1)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return None
    return float(np.median(distances))


def blend_homographies(
    previous: np.ndarray,
    current: np.ndarray,
    alpha: float,
    frame_shape: Sequence[int],
    pitch_length: float,
    pitch_width: float,
    grid_size: int = 6,
) -> np.ndarray:
    """Move ``previous`` towards ``current`` by ``alpha`` in pitch space.

    A grid of image points is projected through both matrices, the pitch
    positions are linearly blended, and a homography is refitted.  Only grid
    points that land on (a margin around) the pitch under both matrices are
    used, so points above the horizon cannot poison the fit.  Falls back to
    ``current`` when too few points are usable.
    """

    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return np.asarray(current, dtype=np.float64)
    height, width = int(frame_shape[0]), int(frame_shape[1])
    xs = np.linspace(0.0, width - 1.0, grid_size)
    ys = np.linspace(0.0, height - 1.0, grid_size)
    image_points = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    def project(matrix: np.ndarray) -> np.ndarray:
        homogeneous = np.hstack([image_points, np.ones((len(image_points), 1))]) @ np.asarray(
            matrix, dtype=np.float64
        ).T
        with np.errstate(divide="ignore", invalid="ignore"):
            return homogeneous[:, :2] / homogeneous[:, 2:3], homogeneous[:, 2]

    previous_world, previous_w = project(previous)
    current_world, current_w = project(current)
    margin_x = 0.5 * pitch_length
    margin_y = 0.5 * pitch_width

    def usable(world: np.ndarray, w: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(world).all(axis=1)
            & (w > 0)
            & (world[:, 0] >= -margin_x)
            & (world[:, 0] <= pitch_length + margin_x)
            & (world[:, 1] >= -margin_y)
            & (world[:, 1] <= pitch_width + margin_y)
        )

    mask = usable(previous_world, previous_w) & usable(current_world, current_w)
    if int(np.count_nonzero(mask)) < MIN_KEYPOINTS_FOR_HOMOGRAPHY + 2:
        return np.asarray(current, dtype=np.float64)
    blended_world = (1.0 - alpha) * previous_world[mask] + alpha * current_world[mask]
    blended, _ = cv2.findHomography(
        image_points[mask].astype(np.float32),
        blended_world.astype(np.float32),
        0,
    )
    if blended is None or not np.isfinite(blended).all():
        return np.asarray(current, dtype=np.float64)
    return blended


class PitchProjectionEngine:
    def __init__(
        self,
        config: SoccerPitchConfiguration,
        fps: float,
        min_keypoint_confidence: float = MIN_KEYPOINT_CONFIDENCE,
        max_reprojection_error_px: float = MAX_REPROJECTION_ERROR_PX,
    ) -> None:
        self.config = config
        self.references = build_pitch_point_references(config)
        self.reference_by_label = {
            reference.label: reference for reference in self.references
        }
        self.references_by_index = {
            reference.index: reference for reference in self.references
        }
        self.min_keypoint_confidence = min_keypoint_confidence
        self.max_reprojection_error_px = max_reprojection_error_px
        self.max_stale_frames = max(1, int(round(max(fps, 1.0) * MAX_STALE_SECONDS)))

        self.prev_valid_homography: Optional[np.ndarray] = None
        self.stale_frames = 0

    def invalidate(self) -> None:
        """Drop the previous matrix after confirmed camera motion."""

        self.prev_valid_homography = None
        self.stale_frames = 0

    def update(
        self,
        frame: np.ndarray,
        keypoints: sv.KeyPoints,
    ) -> PitchProjectionResult:
        observations = self._extract_model_observations(frame, keypoints)

        homography, inlier_labels, reprojection_error = self._estimate_homography(
            observations
        )

        homography_status = 'unavailable'
        projected_keypoints: List[ProjectedPitchKeypoint] = []
        display_tracking = list(observations.values())

        if homography is not None:
            homography_status = 'fresh'
            self.prev_valid_homography = homography
            self.stale_frames = 0
            inlier_set = set(inlier_labels)
            display_tracking = [
                observation
                for observation in observations.values()
                if observation.reference.label in inlier_set
            ]
            projected_keypoints = self._project_keypoints(
                homography,
                [
                    observations[label]
                    for label in inlier_labels
                ],
            )
        elif (
            self.prev_valid_homography is not None
            and self.stale_frames < self.max_stale_frames
        ):
            homography = self.prev_valid_homography
            self.stale_frames += 1
            homography_status = 'stale'
            reprojection_error = None
            projected_keypoints = []
        else:
            self.prev_valid_homography = None
            self.stale_frames = 0

        return PitchProjectionResult(
            tracking_observations=display_tracking,
            projected_keypoints=projected_keypoints,
            homography=homography,
            homography_status=homography_status,
            reprojection_error=reprojection_error,
        )

    def reuse(self, frame: np.ndarray) -> PitchProjectionResult:
        """Reuse the last valid homography for a scheduled-skip frame.

        Unlike :meth:`update`, this method intentionally does not attempt to
        extract or fit keypoints.  The reuse budget is still measured in
        frames, so a fixed camera cannot silently keep an old calibration
        forever.
        """
        if (
            self.prev_valid_homography is None
            or self.stale_frames >= self.max_stale_frames
        ):
            self.prev_valid_homography = None
            self.stale_frames = 0
            return PitchProjectionResult(
                tracking_observations=[],
                projected_keypoints=[],
                homography=None,
                homography_status='unavailable',
                reprojection_error=None,
            )

        self.stale_frames += 1
        return PitchProjectionResult(
            tracking_observations=[],
            projected_keypoints=[],
            homography=self.prev_valid_homography,
            homography_status='reused',
            reprojection_error=None,
        )

    def _extract_model_observations(
        self,
        frame: np.ndarray,
        keypoints: sv.KeyPoints,
    ) -> Dict[str, PitchKeypointObservation]:
        if keypoints.xy.size == 0:
            return {}

        xy = keypoints.xy[0]
        if keypoints.confidence is None:
            confidence = np.ones(xy.shape[0], dtype=np.float32)
        else:
            confidence = keypoints.confidence[0]

        observations: Dict[str, PitchKeypointObservation] = {}
        for index, point in enumerate(xy):
            reference = self.references_by_index.get(index)
            if reference is None:
                continue
            x_value = float(point[0])
            y_value = float(point[1])
            if x_value <= 1 or y_value <= 1:
                continue
            if confidence[index] < self.min_keypoint_confidence:
                continue
            if not self._is_inside_visible_frame(
                (x_value, y_value),
                frame.shape,
                margin=FRAME_EDGE_MARGIN_PX,
            ):
                continue

            observations[reference.label] = PitchKeypointObservation(
                reference=reference,
                image_xy=(x_value, y_value),
                confidence=float(confidence[index]),
                source='model',
            )
        return observations

    def _estimate_homography(
        self,
        observations: Dict[str, PitchKeypointObservation],
    ) -> Tuple[Optional[np.ndarray], List[str], Optional[float]]:
        if len(observations) < MIN_KEYPOINTS_FOR_HOMOGRAPHY:
            return None, [], None

        ordered = list(observations.values())
        image_points = np.array(
            [observation.image_xy for observation in ordered],
            dtype=np.float32,
        )
        world_points = np.array(
            [observation.reference.world_xy for observation in ordered],
            dtype=np.float32,
        )
        minimum_inliers = (
            MIN_INLIERS_FOR_HOMOGRAPHY
            if len(ordered) >= MIN_INLIERS_FOR_HOMOGRAPHY
            else len(ordered)
        )

        for method in (cv2.RANSAC, cv2.RHO, cv2.LMEDS):
            try:
                homography, mask = cv2.findHomography(
                    image_points,
                    world_points,
                    method,
                    5.0,
                )
            except cv2.error:
                homography = None
                mask = None

            if homography is None:
                continue

            inlier_mask = (
                np.ones(len(ordered), dtype=bool)
                if mask is None
                else mask.flatten().astype(bool)
            )
            if int(np.count_nonzero(inlier_mask)) < max(
                MIN_KEYPOINTS_FOR_HOMOGRAPHY,
                minimum_inliers,
            ):
                continue

            try:
                inverse_homography = np.linalg.inv(homography)
            except np.linalg.LinAlgError:
                continue

            reprojected = cv2.perspectiveTransform(
                world_points.reshape(-1, 1, 2),
                inverse_homography,
            ).reshape(-1, 2)
            reprojection_error = np.linalg.norm(reprojected - image_points, axis=1)
            mean_error = float(np.mean(reprojection_error[inlier_mask]))
            if mean_error > self.max_reprojection_error_px:
                continue

            inlier_labels = [
                ordered[index].reference.label
                for index, is_inlier in enumerate(inlier_mask)
                if is_inlier
            ]
            return homography, inlier_labels, mean_error

        return None, [], None

    def _project_keypoints(
        self,
        homography: np.ndarray,
        observations: Sequence[PitchKeypointObservation],
    ) -> List[ProjectedPitchKeypoint]:
        if len(observations) == 0:
            return []
        image_points = np.array(
            [observation.image_xy for observation in observations],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        projected_points = cv2.perspectiveTransform(image_points, homography).reshape(-1, 2)
        return [
            ProjectedPitchKeypoint(
                reference=observation.reference,
                projected_world_xy=(float(projected_points[index][0]), float(projected_points[index][1])),
                source=observation.source,
            )
            for index, observation in enumerate(observations)
        ]

    def _group_reference_labels(self, axis: int) -> Dict[float, List[str]]:
        grouped: Dict[float, List[str]] = {}
        for reference in self.references:
            value = float(reference.world_xy[axis])
            grouped.setdefault(value, []).append(reference.label)
        return grouped

    def _is_inside_visible_frame(
        self,
        point: Tuple[float, float],
        frame_shape: Sequence[int],
        margin: int,
    ) -> bool:
        height = int(frame_shape[0])
        width = int(frame_shape[1])
        return (
            margin <= point[0] < width - margin
            and margin <= point[1] < height - margin
        )
