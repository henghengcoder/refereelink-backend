from __future__ import annotations

import numpy as np

from app.geometry.position_filter import PlayerPositionFilter


def _run(filter_: PlayerPositionFilter, points: list[tuple[float, float]], start: int = 1):
    outputs = []
    for offset, point in enumerate(points):
        out = filter_.update([7], np.array([point], dtype=np.float32), start + offset, [60.0])
        outputs.append(out[0].astype(float))
    return np.array(outputs)


def test_static_player_noise_is_reduced() -> None:
    rng = np.random.default_rng(0)
    truth = np.array([5000.0, 3000.0])
    noisy = truth + rng.normal(0.0, 40.0, size=(200, 2))

    filtered = _run(PlayerPositionFilter(fps=25), [tuple(p) for p in noisy])

    raw_error = np.linalg.norm(noisy[50:] - truth, axis=1).mean()
    filtered_error = np.linalg.norm(filtered[50:] - truth, axis=1).mean()
    assert filtered_error < raw_error * 0.5


def test_first_measurement_is_returned_unchanged() -> None:
    out = PlayerPositionFilter().update([1], np.array([[1234.0, 567.0]]), 10, [80.0])

    assert np.allclose(out, [[1234.0, 567.0]])


def test_single_frame_jump_is_suppressed() -> None:
    filter_ = PlayerPositionFilter(fps=25)
    points = [(5000.0, 3000.0)] * 20 + [(5300.0, 3000.0)] + [(5000.0, 3000.0)] * 2

    filtered = _run(filter_, points)

    assert abs(filtered[20][0] - 5000.0) < 20.0
    assert filter_.outliers_rejected == 1


def test_persistent_jump_resets_after_streak() -> None:
    filter_ = PlayerPositionFilter(fps=25, max_outlier_streak=3)
    points = [(5000.0, 3000.0)] * 20 + [(7000.0, 3000.0)] * 3

    filtered = _run(filter_, points)

    assert abs(filtered[21][0] - 5000.0) < 20.0
    assert np.allclose(filtered[22], [7000.0, 3000.0])
    assert filter_.resets == 1


def test_constant_velocity_is_tracked_without_large_lag() -> None:
    # 7 m/s sprint along x: 28 cm per frame at 25 fps.
    points = [(1000.0 + 28.0 * i, 3000.0) for i in range(100)]

    filtered = _run(PlayerPositionFilter(fps=25), points)

    lag_cm = np.abs(filtered[60:, 0] - np.array(points)[60:, 0]).max()
    assert lag_cm < 30.0


def test_missing_measurement_is_not_invented() -> None:
    filter_ = PlayerPositionFilter()
    filter_.update([1, 2], np.array([[1000.0, 1000.0], [2000.0, 2000.0]]), 1, [60.0, 60.0])

    out = filter_.update(
        [1, 2], np.array([[1010.0, 1000.0], [np.nan, np.nan]]), 2, [60.0, 60.0]
    )

    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_stale_tracks_expire() -> None:
    filter_ = PlayerPositionFilter(max_missing_frames=5)
    filter_.update([1], np.array([[1000.0, 1000.0]]), 1, [60.0])
    filter_.update([2], np.array([[2000.0, 2000.0]]), 10, [60.0])

    assert filter_.active_keys == {2}
