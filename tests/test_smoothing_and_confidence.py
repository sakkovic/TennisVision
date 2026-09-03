"""Tests for temporal smoothing, spike rejection and the confidence policy."""

from __future__ import annotations

import numpy as np
import pytest

from src.measurement import (
    Measurement, MeasurementStatus, build_measurement, classify_confidence,
    summarise, summarise_angular,
)
from src.pose import landmarks as L
from src.pose.smoothing import (
    OneEuroFilter, detect_spikes, find_runs, interpolate_short_gaps,
    oneeuro_smooth_signal, reject_spikes, savgol_smooth_signal, smooth_sequence,
)

from conftest import make_sequence


# ---------------------------------------------------------------------------
# confidence policy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "confidence, expected",
    [
        (0.95, MeasurementStatus.MEASURED),
        (0.50, MeasurementStatus.MEASURED),          # threshold is inclusive
        (0.49, MeasurementStatus.LOW_CONFIDENCE),
        (0.30, MeasurementStatus.LOW_CONFIDENCE),    # floor is inclusive
        (0.29, MeasurementStatus.UNAVAILABLE),
        (0.00, MeasurementStatus.UNAVAILABLE),
        (float("nan"), MeasurementStatus.UNAVAILABLE),
    ],
)
def test_confidence_classification_boundaries(confidence, expected):
    assert classify_confidence(confidence, 0.5, 0.3) is expected


def test_unavailable_measurements_carry_no_number():
    measurement = build_measurement(120.0, 0.1, 0.5, 0.3, "deg")
    assert measurement.status is MeasurementStatus.UNAVAILABLE
    assert measurement.value is None
    assert np.isnan(measurement.as_float())


def test_a_non_finite_value_can_never_be_reported():
    """Even at perfect confidence, NaN geometry must not become a number."""
    measurement = build_measurement(float("nan"), 1.0, 0.5, 0.3, "deg")
    assert measurement.status is MeasurementStatus.UNAVAILABLE
    assert measurement.value is None


def test_summary_uses_only_measured_samples():
    series = [
        build_measurement(10.0, 0.9, 0.5, 0.3, "deg"),
        build_measurement(20.0, 0.9, 0.5, 0.3, "deg"),
        build_measurement(1000.0, 0.4, 0.5, 0.3, "deg"),    # low confidence
        build_measurement(9999.0, 0.1, 0.5, 0.3, "deg"),    # unavailable
    ]
    stats = summarise(series)
    assert stats["mean"] == pytest.approx(15.0)
    assert stats["max"] == pytest.approx(20.0)
    assert stats["frames_measured"] == 2
    assert stats["frames_low_confidence"] == 1
    assert stats["frames_unavailable"] == 1
    assert stats["coverage_percentage"] == pytest.approx(50.0)


def test_summary_without_any_measured_sample_reports_nothing():
    stats = summarise([build_measurement(10.0, 0.1, 0.5, 0.3, "deg")])
    assert stats["status"] == MeasurementStatus.UNAVAILABLE.value
    assert stats["mean"] is None


def test_circular_mean_beats_the_arithmetic_mean_across_the_branch_cut():
    series = [build_measurement(v, 0.9, 0.5, 0.3, "deg") for v in (179.0, -179.0)]
    linear = summarise(series)
    circular = summarise_angular(series)
    assert linear["mean"] == pytest.approx(0.0)             # the wrong answer
    assert abs(circular["mean"]) == pytest.approx(180.0)    # the right one
    assert circular["range"] == pytest.approx(2.0, abs=0.01)
    # Two samples 2 degrees apart are almost perfectly concentrated:
    # R = cos(1 degree) = 0.99985.
    assert circular["resultant_length"] == pytest.approx(np.cos(np.radians(1.0)), abs=1e-3)


def test_circular_and_linear_agree_away_from_the_cut():
    series = [build_measurement(v, 0.9, 0.5, 0.3, "deg") for v in (10.0, 20.0, 30.0)]
    assert summarise_angular(series)["mean"] == pytest.approx(summarise(series)["mean"], abs=1e-6)


def test_resultant_length_falls_when_the_body_sweeps_widely():
    wide = [build_measurement(v, 0.9, 0.5, 0.3, "deg") for v in np.linspace(-170, 170, 20)]
    narrow = [build_measurement(v, 0.9, 0.5, 0.3, "deg") for v in np.linspace(10, 20, 20)]
    assert summarise_angular(wide)["resultant_length"] < 0.2
    assert summarise_angular(narrow)["resultant_length"] > 0.99


# ---------------------------------------------------------------------------
# gaps
# ---------------------------------------------------------------------------
def test_find_runs():
    assert find_runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert find_runs(np.array([1, 1], dtype=bool)) == [(0, 2)]
    assert find_runs(np.zeros(3, dtype=bool)) == []


def test_short_gaps_are_bridged_and_long_ones_are_not():
    signal = np.array([0.0, 1.0, np.nan, 3.0, 4.0] + [np.nan] * 6 + [11.0, 12.0])
    out, filled = interpolate_short_gaps(signal, max_gap=5)
    assert out[2] == pytest.approx(2.0)
    assert filled[2]
    assert np.isnan(out[5:11]).all()          # 6 > max_gap, left alone
    assert not filled[5:11].any()


def test_leading_and_trailing_gaps_are_never_extrapolated():
    signal = np.array([np.nan, np.nan, 2.0, 3.0, np.nan])
    out, filled = interpolate_short_gaps(signal, max_gap=5)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[4])
    assert not filled.any()


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------
def test_savgol_reduces_noise():
    t = np.linspace(0, 2 * np.pi, 60)
    clean = np.sin(t) * 100
    noisy = clean + np.random.default_rng(0).normal(0, 3, 60)
    smoothed = savgol_smooth_signal(noisy, 9, 2)
    assert np.sqrt(np.mean((smoothed - clean) ** 2)) < np.sqrt(np.mean((noisy - clean) ** 2))


def test_savgol_preserves_a_polynomial_exactly():
    """A quadratic is in the filter's model space, so it must pass through."""
    t = np.arange(40, dtype=float)
    quadratic = 3.0 * t ** 2 - 5.0 * t + 7.0
    assert savgol_smooth_signal(quadratic, 9, 2) == pytest.approx(quadratic, rel=1e-6)


def test_savgol_leaves_runs_too_short_to_fit_untouched():
    short = np.array([np.nan, 1.0, 2.0, np.nan])
    out = savgol_smooth_signal(short, 9, 2)
    assert out[1] == 1.0 and out[2] == 2.0


def test_one_euro_passes_a_constant_through():
    constant = np.full(20, 5.0)
    assert oneeuro_smooth_signal(constant, 30.0, 1.0, 0.0, 1.0) == pytest.approx(constant)


def test_one_euro_first_sample_is_unchanged():
    filt = OneEuroFilter(freq=30.0)
    assert filt(3.5) == pytest.approx(3.5)


def test_one_euro_rejects_a_nonsensical_frequency():
    with pytest.raises(ValueError):
        OneEuroFilter(freq=0.0)


# ---------------------------------------------------------------------------
# spike rejection
# ---------------------------------------------------------------------------
def test_detects_a_single_frame_excursion():
    track = np.stack([np.arange(20, dtype=float), np.zeros(20)], axis=1)
    track[10, 0] += 90.0                       # jump out and straight back
    spikes = detect_spikes(track)
    assert spikes[10]
    assert spikes.sum() == 1


def test_genuine_fast_movement_is_not_a_spike():
    """A real acceleration keeps going; it does not return to where it was."""
    track = np.stack([np.concatenate([np.arange(10, dtype=float),
                                      np.arange(10, dtype=float) * 40 + 10]),
                      np.zeros(20)], axis=1)
    assert not detect_spikes(track).any()


def test_a_stationary_landmark_is_not_all_spikes():
    rng = np.random.default_rng(3)
    track = rng.normal(0, 0.3, size=(30, 2)) + 100.0     # tiny jitter only
    assert detect_spikes(track).sum() == 0


def test_reject_spikes_blanks_every_coordinate_system():
    sequence = make_sequence(n_frames=20)
    index = L.IDX["right_wrist"]
    sequence.image_xy[10, index, 0] += 200.0
    sequence.world_xyz[10, index, 0] += 1.0

    cleaned, mask = reject_spikes(sequence)
    assert mask[10, index]
    assert np.isnan(cleaned.image_xy[10, index]).all()
    assert np.isnan(cleaned.world_xyz[10, index]).all()
    assert np.isnan(cleaned.normalized_xyz[10, index]).all()
    # Untouched landmarks survive intact.
    assert np.isfinite(cleaned.image_xy[10, L.IDX["left_wrist"]]).all()


def test_a_repaired_landmark_can_never_be_reported_as_measured(config):
    """
    The point of spike rejection is honesty, not cosmetics: once a coordinate
    has been synthesised, metrics built on it must be downgraded.
    """
    sequence = make_sequence(n_frames=20, visibility=0.99)
    index = L.IDX["right_wrist"]
    sequence.image_xy[10, index, 0] += 200.0

    smoothed, report = smooth_sequence(sequence, config)
    assert report.spike_samples_removed >= 1
    assert smoothed.visibility[10, index] < config.confidence_threshold
    assert smoothed.visibility[9, index] == pytest.approx(0.99)


def test_smoothing_leaves_detection_flags_and_confidence_alone(config):
    sequence = make_sequence(n_frames=20)
    sequence.detected[7] = False
    sequence.image_xy[7] = np.nan
    sequence.world_xyz[7] = np.nan
    sequence.visibility[7] = 0.0

    smoothed, report = smooth_sequence(sequence, config)
    assert smoothed.detected[7] == np.False_
    assert smoothed.visibility[7].max() == 0.0
    assert report.interpolated_frames >= 1
    assert 7 in report.interpolated_frame_indices


def test_smoothing_none_is_a_passthrough(config):
    config.smoothing_method = "none"
    config.spike_rejection = False
    sequence = make_sequence(n_frames=12)
    smoothed, report = smooth_sequence(sequence, config)
    assert report.method == "none"
    assert smoothed.image_xy == pytest.approx(sequence.image_xy)


def test_both_filters_run_over_a_whole_sequence(config):
    for method in ("savgol", "oneeuro"):
        config.smoothing_method = method
        smoothed, report = smooth_sequence(make_sequence(n_frames=25), config)
        assert report.method == method
        assert np.isfinite(smoothed.image_xy).all()
