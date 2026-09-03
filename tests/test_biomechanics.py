"""Known-answer tests for the biomechanical calculations."""

from __future__ import annotations

import numpy as np
import pytest

from src.biomechanics.angles import angle, angle_series, compute_joint_angles
from src.biomechanics.normalization import compute_body_scale
from src.biomechanics.rotations import (
    compute_rotations, segment_tilt_series, segment_yaw_series,
    torso_inclination_series, torso_lateral_lean_series, unwrap_degrees,
    wrap_to_90, wrap_to_180,
)
from src.biomechanics.trajectories import build_trajectories, derivative_per_run
from src.measurement import MeasurementStatus
from src.pose import landmarks as L

from conftest import make_sequence


# ---------------------------------------------------------------------------
# angle()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "a, b, c, expected",
    [
        ([0, 1], [0, 0], [1, 0], 90.0),
        ([-1, 0], [0, 0], [1, 0], 180.0),
        ([1, 0], [0, 0], [1, 0], 0.0),
        ([1, 1], [0, 0], [1, 0], 45.0),
        ([0, 0, 1], [0, 0, 0], [1, 0, 0], 90.0),
        ([1, 0, 0], [0, 0, 0], [0.5, np.sqrt(3) / 2, 0], 60.0),
    ],
)
def test_angle_known_values(a, b, c, expected):
    assert angle(a, b, c) == pytest.approx(expected, abs=1e-6)


def test_angle_is_symmetric_in_its_outer_points():
    assert angle([2, 3], [0, 0], [4, 1]) == pytest.approx(angle([4, 1], [0, 0], [2, 3]))


def test_angle_is_scale_invariant():
    small = angle([0, 1], [0, 0], [1, 0])
    large = angle([0, 1000], [0, 0], [1000, 0])
    assert small == pytest.approx(large)


def test_angle_stable_at_the_extremes():
    """arccos-based formulas lose precision here; atan2 must not."""
    assert angle([-1, 1e-12], [0, 0], [1, 0]) == pytest.approx(180.0, abs=1e-3)
    assert angle([1, 1e-12], [0, 0], [1, 0]) == pytest.approx(0.0, abs=1e-3)


def test_angle_rejects_missing_and_degenerate_input():
    assert np.isnan(angle([np.nan, 0], [0, 0], [1, 0]))
    assert np.isnan(angle([0, 0], [0, 0], [1, 0]))          # coincident points
    assert np.isnan(angle([1, 0], [0, 0], [0, 0]))


def test_angle_series_matches_the_scalar_function():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(8, L.NUM_LANDMARKS, 3)) * 100
    series = angle_series(coords, "right_shoulder", "right_elbow", "right_wrist")
    manual = [
        angle(coords[i, L.IDX["right_shoulder"]], coords[i, L.IDX["right_elbow"]],
              coords[i, L.IDX["right_wrist"]])
        for i in range(8)
    ]
    assert np.allclose(series, manual)


# ---------------------------------------------------------------------------
# joint angles on a known body
# ---------------------------------------------------------------------------
def test_joint_angles_on_synthetic_body(config):
    angles = compute_joint_angles(make_sequence(), config)
    assert angles["right_elbow"][0].value == pytest.approx(90.0, abs=0.5)
    assert angles["left_elbow"][0].value == pytest.approx(90.0, abs=0.5)
    assert angles["right_knee"][0].value == pytest.approx(180.0, abs=0.5)
    assert angles["left_knee"][0].value == pytest.approx(180.0, abs=0.5)
    assert all(m.status is MeasurementStatus.MEASURED for m in angles["right_elbow"])


def test_low_visibility_downgrades_but_keeps_the_value(config):
    angles = compute_joint_angles(make_sequence(visibility=0.4), config)
    measurement = angles["right_elbow"][0]
    assert measurement.status is MeasurementStatus.LOW_CONFIDENCE
    assert measurement.value == pytest.approx(90.0, abs=0.5)


def test_visibility_below_the_floor_yields_no_number(config):
    angles = compute_joint_angles(make_sequence(visibility=0.1), config)
    measurement = angles["right_elbow"][0]
    assert measurement.status is MeasurementStatus.UNAVAILABLE
    assert measurement.value is None


def test_undetected_frames_produce_no_measurements(config):
    sequence = make_sequence()
    sequence.detected[5] = False
    sequence.world_xyz[5] = np.nan
    sequence.visibility[5] = 0.0
    angles = compute_joint_angles(sequence, config)
    assert angles["right_elbow"][5].status is MeasurementStatus.UNAVAILABLE
    assert angles["right_elbow"][5].value is None
    assert angles["right_elbow"][4].status is MeasurementStatus.MEASURED


def test_a_single_weak_landmark_downgrades_the_whole_angle(config):
    """An occluded wrist must take the elbow angle down with it."""
    sequence = make_sequence()
    sequence.visibility[:, L.IDX["right_wrist"]] = 0.35
    angles = compute_joint_angles(sequence, config)
    assert angles["right_elbow"][0].status is MeasurementStatus.LOW_CONFIDENCE
    assert angles["left_elbow"][0].status is MeasurementStatus.MEASURED


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------
def test_wrapping_helpers():
    assert wrap_to_180(190) == pytest.approx(-170)
    assert wrap_to_180(-190) == pytest.approx(170)
    assert wrap_to_90(170) == pytest.approx(-10)
    assert wrap_to_90(-170) == pytest.approx(10)


def _yaw_frame(left, right):
    world = np.full((1, L.NUM_LANDMARKS, 3), np.nan)
    world[0, L.IDX["left_shoulder"]] = left
    world[0, L.IDX["right_shoulder"]] = right
    return world


@pytest.mark.parametrize(
    "left, right, expected, description",
    [
        ((-0.2, -0.3, 0.0), (0.2, -0.3, 0.0), 0.0, "square to camera, facing away"),
        ((0.2, -0.3, 0.0), (-0.2, -0.3, 0.0), 180.0, "square to camera, facing it"),
        ((0.0, -0.3, 0.2), (0.0, -0.3, -0.2), -90.0, "side-on, right shoulder nearer"),
        ((0.0, -0.3, -0.2), (0.0, -0.3, 0.2), 90.0, "side-on, left shoulder nearer"),
        ((-0.2, -0.3, -0.2), (0.2, -0.3, 0.2), 45.0, "half turned"),
    ],
)
def test_segment_yaw_conventions(left, right, expected, description):
    value = float(segment_yaw_series(_yaw_frame(left, right),
                                     "left_shoulder", "right_shoulder")[0])
    assert value == pytest.approx(expected, abs=0.01), description


def test_torso_inclination_and_lean(config):
    sequence = make_sequence()
    rotations = compute_rotations(sequence, config)
    assert rotations["torso_inclination"][0].value == pytest.approx(0.0, abs=0.5)
    assert rotations["torso_lateral_lean"][0].value == pytest.approx(0.0, abs=0.5)
    assert rotations["shoulder_hip_separation"][0].value == pytest.approx(0.0, abs=0.5)


def test_lateral_lean_sign_points_right():
    world = np.zeros((1, L.NUM_LANDMARKS, 3))
    offset = np.tan(np.radians(30.0)) * 0.6
    world[0, L.IDX["left_shoulder"]] = (-0.2 + offset, -0.3, 0.0)
    world[0, L.IDX["right_shoulder"]] = (0.2 + offset, -0.3, 0.0)
    world[0, L.IDX["left_hip"]] = (-0.1, 0.3, 0.0)
    world[0, L.IDX["right_hip"]] = (0.1, 0.3, 0.0)
    assert float(torso_inclination_series(world)[0]) == pytest.approx(30.0, abs=0.01)
    assert float(torso_lateral_lean_series(world)[0]) == pytest.approx(30.0, abs=0.01)


def test_image_tilt_is_positive_when_the_right_side_is_lower():
    image = np.full((1, L.NUM_LANDMARKS, 2), np.nan)
    image[0, L.IDX["left_shoulder"]] = (100, 100)
    image[0, L.IDX["right_shoulder"]] = (200, 200)
    tilt = float(segment_tilt_series(image, "left_shoulder", "right_shoulder")[0])
    assert tilt == pytest.approx(45.0, abs=0.01)


def test_unwrap_bridges_the_branch_cut_but_not_gaps():
    series = np.array([170.0, 175.0, -179.0, -172.0, np.nan, 80.0, 85.0])
    out = unwrap_degrees(series)
    assert out[:4] == pytest.approx([170.0, 175.0, 181.0, 188.0])
    assert np.isnan(out[4])
    assert out[5:] == pytest.approx([80.0, 85.0])       # run after the gap left alone


def test_unwrap_handles_degenerate_input():
    assert np.isnan(unwrap_degrees(np.array([np.nan, np.nan]))).all()
    assert unwrap_degrees(np.array([np.nan, 5.0, np.nan]))[1] == 5.0


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------
def test_derivative_recovers_constant_velocity():
    t = np.arange(10) / 10.0
    track = np.stack([t * 50.0, np.zeros(10)], axis=1)
    assert derivative_per_run(track, t)[:, 0] == pytest.approx(np.full(10, 50.0))


def test_derivative_does_not_bridge_a_gap():
    t = np.arange(10) / 10.0
    track = np.stack([t * 50.0, np.zeros(10)], axis=1)
    track[4:6] = np.nan
    out = derivative_per_run(track, t)
    assert np.isnan(out[4:6]).all()
    assert out[0, 0] == pytest.approx(50.0)
    assert out[9, 0] == pytest.approx(50.0)


def test_isolated_sample_has_no_velocity():
    track = np.full((5, 2), np.nan)
    track[2] = (1.0, 1.0)
    assert np.isnan(derivative_per_run(track, np.arange(5) / 10.0)).all()


def test_trajectory_path_length_skips_unmeasured_frames(config):
    sequence = make_sequence(n_frames=10)
    # Move the hips 10 px per frame along x.
    for i in range(10):
        sequence.image_xy[i, L.IDX["left_hip"], 0] += 10 * i
        sequence.image_xy[i, L.IDX["right_hip"], 0] += 10 * i

    trajectories = build_trajectories(sequence, config)
    summary = trajectories["hip_center"].summary(compute_body_scale(sequence, config))
    assert summary["path_length_px"] == pytest.approx(90.0, abs=1e-6)
    assert summary["skipped_transitions"] == 0

    sequence.visibility[5, L.IDX["left_hip"]] = 0.1       # drop one frame out
    trajectories = build_trajectories(sequence, config)
    summary = trajectories["hip_center"].summary(compute_body_scale(sequence, config))
    assert summary["skipped_transitions"] == 1
    assert summary["path_length_is_lower_bound"] is True
    assert summary["path_length_px"] == pytest.approx(70.0, abs=1e-6)


def test_trajectory_reports_unavailable_without_enough_data(config):
    sequence = make_sequence(n_frames=10, visibility=0.05)
    trajectories = build_trajectories(sequence, config)
    summary = trajectories["hip_center"].summary(compute_body_scale(sequence, config))
    assert summary["status"] == MeasurementStatus.UNAVAILABLE.value
    assert "path_length_px" not in summary


# ---------------------------------------------------------------------------
# scale normalisation
# ---------------------------------------------------------------------------
def test_body_scale_uses_the_torso(config):
    sequence = make_sequence()
    scale = compute_body_scale(sequence, config)
    # 0.50 m of trunk at 200 px/m.
    assert scale.torso_length_px == pytest.approx(100.0, abs=0.5)
    assert scale.shoulder_width_px == pytest.approx(80.0, abs=0.5)
    assert scale.has_body_units
    assert not scale.has_metric_estimate            # no height supplied
    assert scale.px_to_body_units(50.0) == pytest.approx(0.5)


def test_metric_scale_only_appears_with_a_player_height(config):
    config.player_height_m = 1.80
    scale = compute_body_scale(make_sequence(), config)
    assert scale.has_metric_estimate
    # 0.288 * 1.80 m spread over 100 px.
    assert scale.metres_per_pixel == pytest.approx(0.288 * 1.80 / 100.0, rel=1e-6)


def test_no_confident_torso_means_no_scale(config):
    scale = compute_body_scale(make_sequence(visibility=0.1), config)
    assert not scale.has_body_units
    assert np.isnan(scale.px_to_body_units(50.0))
