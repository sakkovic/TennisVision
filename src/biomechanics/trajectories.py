"""
Movement trajectories.

A trajectory is the path one point traces through the image over time.  Image
coordinates are used rather than the MediaPipe world landmarks because the
world landmarks are re-centred on the hips in every frame: in that space the
hip centre is the origin by construction and never moves, so global movement is
invisible there.  Image space keeps the movement but carries no absolute scale,
which is what :mod:`src.biomechanics.normalization` addresses.

Speed
-----
Speed is computed from the *smoothed* coordinates, because differentiating raw
landmark jitter produces large spurious velocities.  Central differences are
used inside each contiguous run of valid samples and one-sided differences at
its ends; runs are never joined across a gap, since the displacement over an
unobserved interval is unknown.

Speed is reported in three units:

* ``px_per_s``          raw image speed, valid only within this one clip
* ``body_units_per_s``  divided by the player torso length, comparable between
                        clips and camera distances
* ``m_per_s``           only when ``--player-height`` was supplied; an estimate

All of these measure movement *in the image plane*.  A wrist travelling
straight toward the camera barely moves on screen, so its true speed is
underestimated.  These are not stroke speeds.

Path length
-----------
The sum of frame-to-frame displacements, accumulated only over consecutive
frame pairs where both samples are confidently measured.  Any gap is skipped
rather than bridged, and the number of skipped transitions is reported so the
reader knows the path length is a lower bound when coverage is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..measurement import MeasurementStatus, classify_confidence
from ..pose import landmarks as L
from .normalization import BodyScale

# The points whose trajectories the brief asks for.
TRAJECTORY_POINTS: List[str] = [
    "right_wrist",
    "left_wrist",
    "hip_center",
    "left_foot",
    "right_foot",
]

TRAJECTORY_DESCRIPTIONS: Dict[str, str] = {
    "right_wrist": "Right wrist landmark path in image pixels.",
    "left_wrist": "Left wrist landmark path in image pixels.",
    "hip_center": "Midpoint of the left and right hip landmarks. A proxy for pelvis "
                  "position; NOT the body centre of mass.",
    "left_foot": "Midpoint of the left heel and left foot index (toe) landmarks.",
    "right_foot": "Midpoint of the right heel and right foot index (toe) landmarks.",
}


def derivative_per_run(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Time derivative computed independently on each contiguous run of samples.

    ``values`` may be (T,) or (T, D).  Runs shorter than two samples yield NaN,
    because a single isolated observation carries no velocity information.
    """
    values = np.asarray(values, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)

    valid = np.isfinite(values)
    if values.ndim > 1:
        valid = valid.all(axis=-1)
    valid &= np.isfinite(timestamps)
    if not valid.any():
        return out

    start: Optional[int] = None
    for i in range(len(values) + 1):
        is_valid = bool(valid[i]) if i < len(values) else False
        if is_valid and start is None:
            start = i
        elif not is_valid and start is not None:
            if i - start >= 2:
                t = timestamps[start:i]
                if np.all(np.diff(t) > 0):
                    out[start:i] = np.gradient(values[start:i], t, axis=0)
            start = None
    return out


@dataclass
class Trajectory:
    """One point tracked across the clip, with derived movement statistics."""

    name: str
    frame_indices: np.ndarray
    timestamps: np.ndarray
    xy_px: np.ndarray                 # (T, 2), NaN where UNAVAILABLE
    xy_normalized: np.ndarray         # (T, 2)
    confidence: np.ndarray            # (T,)
    statuses: List[MeasurementStatus] = field(default_factory=list)
    speed_px_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    velocity_px_s: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # -- masks ----------------------------------------------------------
    @property
    def measured_mask(self) -> np.ndarray:
        return np.array([s is MeasurementStatus.MEASURED for s in self.statuses], dtype=bool)

    @property
    def usable_mask(self) -> np.ndarray:
        return np.array(
            [s is not MeasurementStatus.UNAVAILABLE for s in self.statuses], dtype=bool
        )

    def points_for_drawing(self, only_measured: bool = False) -> np.ndarray:
        mask = self.measured_mask if only_measured else self.usable_mask
        out = self.xy_px.copy()
        out[~mask] = np.nan
        return out

    # -- statistics -----------------------------------------------------
    def summary(self, scale: BodyScale) -> Dict:
        mask = self.measured_mask
        pts = self.xy_px[mask]
        n_measured = int(mask.sum())

        base: Dict[str, object] = {
            "description": TRAJECTORY_DESCRIPTIONS.get(self.name, ""),
            "unit": "pixels (image coordinates, origin top-left)",
            "frames_total": int(len(self.statuses)),
            "frames_measured": n_measured,
            "frames_low_confidence": int(
                sum(1 for s in self.statuses if s is MeasurementStatus.LOW_CONFIDENCE)
            ),
            "frames_unavailable": int(
                sum(1 for s in self.statuses if s is MeasurementStatus.UNAVAILABLE)
            ),
            "coverage_percentage": round(100.0 * n_measured / len(self.statuses), 2)
            if self.statuses else 0.0,
        }

        if n_measured < 2:
            base.update({
                "status": MeasurementStatus.UNAVAILABLE.value,
                "note": "Fewer than two confidently measured frames; no path can be described.",
            })
            return base

        # Path length over consecutive measured pairs only.
        idx = np.flatnonzero(mask)
        consecutive = np.diff(idx) == 1
        steps = np.linalg.norm(np.diff(self.xy_px[idx], axis=0), axis=1)
        path_px = float(np.sum(steps[consecutive]))
        skipped = int(np.count_nonzero(~consecutive))

        displacement_vec = self.xy_px[idx[-1]] - self.xy_px[idx[0]]
        displacement_px = float(np.linalg.norm(displacement_vec))

        x0, y0 = np.min(pts, axis=0)
        x1, y1 = np.max(pts, axis=0)

        speeds = self.speed_px_s[mask] if self.speed_px_s.size else np.zeros(0)
        speeds = speeds[np.isfinite(speeds)]

        base.update({
            "status": MeasurementStatus.MEASURED.value,
            "start_px": [round(float(self.xy_px[idx[0]][0]), 2),
                         round(float(self.xy_px[idx[0]][1]), 2)],
            "end_px": [round(float(self.xy_px[idx[-1]][0]), 2),
                       round(float(self.xy_px[idx[-1]][1]), 2)],
            "net_displacement_px": round(displacement_px, 2),
            "net_displacement_vector_px": [round(float(displacement_vec[0]), 2),
                                           round(float(displacement_vec[1]), 2)],
            "path_length_px": round(path_px, 2),
            "path_length_is_lower_bound": skipped > 0,
            "skipped_transitions": skipped,
            "bounding_box_px": {
                "x_min": round(float(x0), 2), "y_min": round(float(y0), 2),
                "x_max": round(float(x1), 2), "y_max": round(float(y1), 2),
                "width": round(float(x1 - x0), 2), "height": round(float(y1 - y0), 2),
            },
            "mean_confidence": round(float(np.mean(self.confidence[mask])), 3),
        })

        if speeds.size:
            peak_index = int(idx[np.nanargmax(np.abs(self.speed_px_s[idx]))]) \
                if np.isfinite(self.speed_px_s[idx]).any() else None
            base["speed_px_per_s"] = {
                "mean": round(float(np.mean(speeds)), 2),
                "max": round(float(np.max(speeds)), 2),
                "median": round(float(np.median(speeds)), 2),
            }
            if peak_index is not None:
                base["peak_speed_frame"] = peak_index
                base["peak_speed_timestamp_s"] = round(float(self.timestamps[peak_index]), 3)

        # Scale-normalised versions.
        if scale.has_body_units:
            base["path_length_body_units"] = round(float(scale.px_to_body_units(path_px)), 3)
            base["net_displacement_body_units"] = round(
                float(scale.px_to_body_units(displacement_px)), 3
            )
            if speeds.size:
                base["speed_body_units_per_s"] = {
                    "mean": round(float(scale.px_to_body_units(np.mean(speeds))), 3),
                    "max": round(float(scale.px_to_body_units(np.max(speeds))), 3),
                }
        if scale.has_metric_estimate:
            base["path_length_m_estimate"] = round(float(scale.px_to_metres(path_px)), 3)
            base["net_displacement_m_estimate"] = round(
                float(scale.px_to_metres(displacement_px)), 3
            )
            if speeds.size:
                base["speed_m_per_s_estimate"] = {
                    "mean": round(float(scale.px_to_metres(np.mean(speeds))), 3),
                    "max": round(float(scale.px_to_metres(np.max(speeds))), 3),
                }
            base["metric_note"] = (
                "Metre values are estimates from an assumed player height and Winter "
                "anthropometric ratios, measured in the image plane only. Not a court-calibrated "
                "measurement of real-world distance or speed."
            )
        return base


def build_trajectory(sequence, point_name: str, config) -> Trajectory:
    """Extract one point trajectory, applying the confidence policy per frame."""
    xy_px = L.point_from_array(sequence.image_xy, point_name).copy()
    xy_norm = L.point_from_array(sequence.normalized_xyz[..., :2], point_name).copy()
    confidence = L.confidence_from_array(sequence.visibility, point_name).copy()

    statuses: List[MeasurementStatus] = []
    for i in range(len(sequence)):
        if not sequence.detected[i] or not np.all(np.isfinite(xy_px[i])):
            statuses.append(MeasurementStatus.UNAVAILABLE)
            xy_px[i] = np.nan
            xy_norm[i] = np.nan
            confidence[i] = 0.0
            continue
        status = classify_confidence(
            float(confidence[i]), config.confidence_threshold, config.low_confidence_floor
        )
        if status is MeasurementStatus.UNAVAILABLE:
            xy_px[i] = np.nan
            xy_norm[i] = np.nan
        statuses.append(status)

    velocity = derivative_per_run(xy_px, sequence.timestamps)
    speed = np.linalg.norm(velocity, axis=-1)

    return Trajectory(
        name=point_name,
        frame_indices=sequence.frame_indices,
        timestamps=sequence.timestamps,
        xy_px=xy_px,
        xy_normalized=xy_norm,
        confidence=confidence,
        statuses=statuses,
        speed_px_s=speed,
        velocity_px_s=velocity,
    )


def resolve_trajectory_points(hand: Optional[str]) -> List[str]:
    """
    Decide which points to track.

    Both wrists are always extracted so the CSV is complete regardless of the
    ``--hand`` flag.  Handedness only decides which wrist is highlighted as the
    dominant one in the report and the overlay; this POC never tries to infer
    handedness from the video.
    """
    return list(TRAJECTORY_POINTS)


def dominant_wrist(hand: Optional[str]) -> Optional[str]:
    if hand == "right":
        return "right_wrist"
    if hand == "left":
        return "left_wrist"
    return None


def build_trajectories(sequence, config) -> Dict[str, Trajectory]:
    """Build every trajectory the report and the overlay need."""
    return {
        name: build_trajectory(sequence, name, config)
        for name in resolve_trajectory_points(config.hand)
    }
