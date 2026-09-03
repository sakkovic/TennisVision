"""
Temporal smoothing of landmark trajectories.

Raw per-frame pose estimates jitter by a few pixels even when the subject is
still.  Differentiating that jitter to get velocity amplifies it badly, so
smoothing has to happen before any trajectory or speed calculation.

Two filters are provided:

``savgol`` (default)
    Savitzky-Golay: fits a low-order polynomial over a sliding window and takes
    the fitted value at the centre.  It is non-causal (it uses future frames),
    which is ideal for offline analysis because it introduces no phase lag and
    preserves the height and timing of peaks, which matters when the quantity
    of interest is a fast swing.

``oneeuro``
    The One Euro filter: a causal, adaptive low-pass filter whose cutoff rises
    with speed, so it removes jitter when the joint is slow and stops lagging
    when the joint is fast.  It is included because it is the filter a future
    real-time version of this product would need.

Both keep the raw arrays untouched: the pipeline carries a raw and a smoothed
:class:`~src.pose.detector.PoseSequence` side by side so they can be compared.

Missing data
------------
Frames where the pose was not detected hold NaN.  Short gaps are bridged by
linear interpolation purely so the filters see a continuous signal; the
``detected`` flag and the visibility scores are NEVER modified.  A frame that
had no detection therefore still reports UNAVAILABLE for every metric even
though its smoothed coordinate exists.  Gaps longer than
``max_interpolation_gap`` are left as NaN and split the signal into independent
runs, each smoothed on its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------
def find_runs(valid: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start, end) index pairs of contiguous True regions."""
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def interpolate_short_gaps(signal: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly bridge NaN gaps of at most ``max_gap`` samples.

    Only *interior* gaps are filled: a leading or trailing run of NaN is left
    alone, because extrapolating outside the observed data would be inventing
    measurements.

    Returns the filled signal and a boolean mask of the samples that were
    synthesised, so callers can keep track of what is real.
    """
    out = signal.astype(float).copy()
    filled = np.zeros(out.shape, dtype=bool)
    valid = np.isfinite(out)
    if valid.all() or not valid.any() or max_gap <= 0:
        return out, filled

    idx = np.arange(out.size)
    first, last = idx[valid][0], idx[valid][-1]
    gap_runs = find_runs(~valid)
    for start, end in gap_runs:
        if start <= first or end > last:      # leading / trailing gap
            continue
        if (end - start) > max_gap:           # too long to trust
            continue
        left, right = out[start - 1], out[end]
        out[start:end] = np.linspace(left, right, end - start + 2)[1:-1]
        filled[start:end] = True
    return out, filled


# ---------------------------------------------------------------------------
# Spike rejection
# ---------------------------------------------------------------------------
def detect_spikes(track_xy: np.ndarray, factor: float = 4.0,
                  min_jump_px: float = 6.0) -> np.ndarray:
    """
    Flag single-frame position outliers in one landmark track.

    Pose models occasionally place a landmark tens of pixels away for exactly
    one frame and then recover, and they do it *while reporting high
    visibility*: the confidence score describes whether the model believes the
    joint is visible, not whether it put it in the right place.  Confidence
    filtering alone therefore cannot catch this, and a polynomial smoother is
    not robust to it either, so one bad sample drags the fitted curve with it.

    A spike is identified geometrically rather than by magnitude alone: the
    point jumps away from its predecessor, jumps back to its successor, and yet
    the predecessor and successor are close to each other.  Genuine fast
    movement fails that last condition, because during real motion the
    surrounding frames are far apart too.  Fast tennis movement is therefore
    preserved and only the there-and-back excursions are removed.

    The threshold adapts to the clip: it is ``factor`` times the median
    frame-to-frame displacement of this landmark, with an absolute floor so
    that a nearly stationary joint does not have every micro-jitter flagged.
    """
    spikes = np.zeros(track_xy.shape[0], dtype=bool)
    finite = np.isfinite(track_xy).all(axis=1)
    if finite.sum() < 3:
        return spikes

    steps = np.linalg.norm(np.diff(track_xy, axis=0), axis=1)
    typical = np.nanmedian(steps[np.isfinite(steps)]) if np.isfinite(steps).any() else 0.0
    threshold = max(min_jump_px, factor * float(typical))

    for i in range(1, len(track_xy) - 1):
        if not (finite[i - 1] and finite[i] and finite[i + 1]):
            continue
        d_prev = float(np.linalg.norm(track_xy[i] - track_xy[i - 1]))
        d_next = float(np.linalg.norm(track_xy[i + 1] - track_xy[i]))
        d_span = float(np.linalg.norm(track_xy[i + 1] - track_xy[i - 1]))
        if d_prev > threshold and d_next > threshold and d_span < 0.5 * (d_prev + d_next):
            spikes[i] = True
    return spikes


def reject_spikes(sequence, factor: float = 4.0) -> tuple["object", np.ndarray]:
    """
    Blank out spiked landmark samples across every coordinate system.

    Returns a copy of the sequence with the offending samples set to NaN (so
    the gap logic bridges them) and a ``(T, 33)`` mask of what was removed.
    The caller is responsible for making sure metrics built on a repaired
    landmark are not advertised as fully measured.
    """
    image_xy = sequence.image_xy.copy()
    normalized = sequence.normalized_xyz.copy()
    world = sequence.world_xyz.copy()
    mask = np.zeros(sequence.visibility.shape, dtype=bool)

    for landmark in range(image_xy.shape[1]):
        spikes = detect_spikes(image_xy[:, landmark, :], factor=factor)
        if not spikes.any():
            continue
        mask[spikes, landmark] = True
        image_xy[spikes, landmark, :] = np.nan
        normalized[spikes, landmark, :] = np.nan
        world[spikes, landmark, :] = np.nan

    return sequence.copy_with(
        image_xy=image_xy, normalized_xyz=normalized, world_xyz=world
    ), mask


# ---------------------------------------------------------------------------
# Savitzky-Golay
# ---------------------------------------------------------------------------
def _odd_at_most(n: int) -> int:
    return n if n % 2 == 1 else n - 1


def savgol_smooth_signal(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    Savitzky-Golay applied independently to each contiguous non-NaN run.

    The window is shrunk to fit short runs.  A run too short to support even a
    minimal polynomial fit is returned unchanged rather than dropped, so no
    data is lost and none is fabricated.
    """
    out = signal.astype(float).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out

    for start, end in find_runs(valid):
        length = end - start
        win = _odd_at_most(min(window, length))
        order = min(polyorder, win - 1) if win > 1 else 0
        if win < 3 or order < 1:
            continue                              # too short: leave raw
        out[start:end] = savgol_filter(out[start:end], window_length=win, polyorder=order)
    return out


# ---------------------------------------------------------------------------
# One Euro
# ---------------------------------------------------------------------------
class OneEuroFilter:
    """
    Scalar One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

    ``min_cutoff`` sets how aggressively slow movement is smoothed; ``beta``
    sets how quickly the filter opens up as speed increases.
    """

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0

    def __call__(self, x: float, dt: Optional[float] = None) -> float:
        dt = (1.0 / self.freq) if dt is None or dt <= 0 else dt
        if self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            return float(x)

        dx = (float(x) - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * float(x) + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def oneeuro_smooth_signal(signal: np.ndarray, freq: float, min_cutoff: float,
                          beta: float, d_cutoff: float) -> np.ndarray:
    """One Euro applied per contiguous run (the filter is reset at each gap)."""
    out = signal.astype(float).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    for start, end in find_runs(valid):
        filt = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        for i in range(start, end):
            out[i] = filt(float(out[i]))
    return out


# ---------------------------------------------------------------------------
# Sequence-level entry point
# ---------------------------------------------------------------------------
@dataclass
class SmoothingReport:
    method: str
    window: Optional[int]
    polyorder: Optional[int]
    max_interpolation_gap: int
    interpolated_frames: int
    interpolated_frame_indices: List[int]
    median_raw_to_smoothed_shift_px: Optional[float]
    max_raw_to_smoothed_shift_px: Optional[float]
    spike_rejection_enabled: bool = False
    spike_samples_removed: int = 0
    spike_landmarks: Dict[str, int] = None

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "window_frames": self.window,
            "polynomial_order": self.polyorder,
            "max_interpolation_gap_frames": self.max_interpolation_gap,
            "frames_with_interpolated_coordinates": self.interpolated_frames,
            "interpolated_frame_indices": self.interpolated_frame_indices,
            "median_shift_px": self.median_raw_to_smoothed_shift_px,
            "max_shift_px": self.max_raw_to_smoothed_shift_px,
            "spike_rejection_enabled": self.spike_rejection_enabled,
            "spike_samples_removed": self.spike_samples_removed,
            "spike_samples_by_landmark": self.spike_landmarks or {},
            "note": (
                "Interpolated coordinates exist only so the filter sees a continuous "
                "signal. Frames with no detection keep detected=False and zero visibility, so "
                "every metric on them is still reported UNAVAILABLE. Landmark samples removed "
                "as position spikes have their confidence capped below the MEASURED threshold, "
                "so any metric depending on a repaired landmark is reported at most as "
                "LOW_CONFIDENCE."
            ),
        }


def _smooth_stack(array: np.ndarray, method: str, fps: float, config,
                  max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth a (T, N, D) landmark stack along the time axis.

    Returns the smoothed stack and a (T,) mask of frames where at least one
    coordinate was synthesised by interpolation.
    """
    T = array.shape[0]
    flat = array.reshape(T, -1)
    out = np.empty_like(flat)
    filled_any = np.zeros(T, dtype=bool)

    for c in range(flat.shape[1]):
        column = flat[:, c]
        bridged, filled = interpolate_short_gaps(column, max_gap)
        filled_any |= filled

        if method == "savgol":
            out[:, c] = savgol_smooth_signal(bridged, config.savgol_window, config.savgol_polyorder)
        elif method == "oneeuro":
            out[:, c] = oneeuro_smooth_signal(
                bridged, fps, config.one_euro_min_cutoff,
                config.one_euro_beta, config.one_euro_d_cutoff,
            )
        else:
            out[:, c] = bridged

    return out.reshape(array.shape), filled_any


def smooth_sequence(sequence, config) -> Tuple["object", SmoothingReport]:
    """
    Produce a smoothed copy of a :class:`~src.pose.detector.PoseSequence`.

    ``detected``, ``visibility`` and ``presence`` are copied through unchanged.
    Only coordinates are filtered.
    """
    from . import landmarks as landmark_module

    method = config.smoothing_method
    fps = sequence.fps if sequence.fps and sequence.fps > 0 else config.fallback_fps
    max_gap = config.max_interpolation_gap

    # --- spike rejection ------------------------------------------------
    spike_mask = np.zeros(sequence.visibility.shape, dtype=bool)
    working = sequence
    if getattr(config, "spike_rejection", True):
        working, spike_mask = reject_spikes(sequence, factor=getattr(config, "spike_factor", 4.0))

    spike_counts = {
        landmark_module.LANDMARK_NAMES[j]: int(spike_mask[:, j].sum())
        for j in range(spike_mask.shape[1]) if spike_mask[:, j].any()
    }
    if spike_counts:
        logger.info(
            "Spike rejection removed %d landmark samples across %d landmarks: %s",
            int(spike_mask.sum()), len(spike_counts), spike_counts,
        )

    # A repaired landmark must never be advertised as fully measured, so its
    # confidence is capped just below the MEASURED threshold for those frames.
    visibility = sequence.visibility.copy()
    if spike_mask.any():
        cap = max(0.0, config.confidence_threshold - 1e-6)
        visibility[spike_mask] = np.minimum(visibility[spike_mask], cap)

    if method == "none":
        report = SmoothingReport(
            method="none", window=None, polyorder=None,
            max_interpolation_gap=0, interpolated_frames=0,
            interpolated_frame_indices=[],
            median_raw_to_smoothed_shift_px=0.0, max_raw_to_smoothed_shift_px=0.0,
            spike_rejection_enabled=bool(getattr(config, "spike_rejection", True)),
            spike_samples_removed=int(spike_mask.sum()), spike_landmarks=spike_counts,
        )
        return working.copy_with(visibility=visibility, source="smoothed(none)"), report

    image_xy, filled_a = _smooth_stack(working.image_xy, method, fps, config, max_gap)
    norm_xyz, filled_b = _smooth_stack(working.normalized_xyz, method, fps, config, max_gap)
    world_xyz, filled_c = _smooth_stack(working.world_xyz, method, fps, config, max_gap)
    filled_any = filled_a | filled_b | filled_c

    # How far did smoothing actually move the joints?  Reported so the effect of
    # the filter is visible and auditable rather than hidden.
    shift = np.linalg.norm(sequence.image_xy - image_xy, axis=-1)
    shift = shift[np.isfinite(shift)]
    median_shift = float(np.median(shift)) if shift.size else None
    max_shift = float(np.max(shift)) if shift.size else None

    smoothed = sequence.copy_with(
        image_xy=image_xy,
        normalized_xyz=norm_xyz,
        world_xyz=world_xyz,
        visibility=visibility,
        source=f"smoothed({method})",
    )
    report = SmoothingReport(
        method=method,
        window=config.savgol_window if method == "savgol" else None,
        polyorder=config.savgol_polyorder if method == "savgol" else None,
        max_interpolation_gap=max_gap,
        interpolated_frames=int(filled_any.sum()),
        interpolated_frame_indices=[int(i) for i in np.flatnonzero(filled_any)],
        median_raw_to_smoothed_shift_px=round(median_shift, 3) if median_shift is not None else None,
        max_raw_to_smoothed_shift_px=round(max_shift, 3) if max_shift is not None else None,
        spike_rejection_enabled=bool(getattr(config, "spike_rejection", True)),
        spike_samples_removed=int(spike_mask.sum()),
        spike_landmarks=spike_counts,
    )
    logger.info(
        "Smoothing (%s): median joint shift %.2f px, max %.2f px, %d frames interpolated",
        method, median_shift or 0.0, max_shift or 0.0, report.interpolated_frames,
    )
    return smoothed, report
