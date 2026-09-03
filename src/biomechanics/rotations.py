"""
Body orientation and rotation estimates.

Coordinate convention
---------------------
All rotation metrics are derived from the MediaPipe **world** landmarks, whose
axes are aligned with the camera:

    +x  to the right of the image
    +y  downwards in the image  (so "up" is -y)
    +z  away from the camera    (MediaPipe documents smaller z as nearer)

The origin is the midpoint of the hips.  Because the origin travels with the
player, world landmarks describe body *shape and orientation* only; they carry
no information about where the player is on the court.  Global movement is
therefore measured from image coordinates instead (see ``trajectories.py``).

Metrics
-------
``shoulder_orientation_deg`` / ``hip_orientation_deg``
    Rotation of the shoulder (or hip) line about the vertical axis, measured
    against the camera image plane, reported in (-180, +180] degrees.

        v = right_landmark - left_landmark
        theta = degrees( atan2(v_z, v_x) )

       0 deg  the line is parallel to the image plane and the player faces
              AWAY from the camera (right landmark on the image right).
    +/-180    parallel to the image plane, player faces TOWARD the camera.
      -90     the line points along the camera axis with the RIGHT landmark
              nearer the camera.
      +90     fully side-on with the LEFT landmark nearer the camera.

    The angle is kept *directed* rather than folded into a half turn.  Folding
    would put the discontinuity at +/-90 degrees, which is precisely where a
    player filmed from the side of the court spends most of the clip, and the
    metric would flip sign every few frames.  Keeping it directed moves the
    branch cut to +/-180 and preserves the information about which shoulder is
    nearer the camera.

    Because any angle has a branch cut somewhere, the time series is also
    published in a temporally unwrapped form (see :func:`unwrap_degrees`).  The
    unwrapped series is what the plots and the "total rotation swept" statistic
    use, so a player turning through the cut produces a continuous curve
    instead of a 360 degree jump.

``shoulder_hip_separation_deg``
    The classic "X-factor": how far the shoulder line is rotated relative to
    the hip line, ``wrap_to_180(shoulder_orientation - hip_orientation)``.  Its
    magnitude is the separation between upper and lower body.  A larger value
    during a backswing means more torso coil relative to the pelvis.  Because
    the two segments belong to one body the difference is naturally small and
    stays well clear of the branch cut.  This POC reports the number only; it
    does not judge whether more or less is better.

``torso_inclination_deg``
    Angle between the trunk vector (hip midpoint -> shoulder midpoint) and true
    vertical, in [0, 180], where 0 means perfectly upright.  It is unsigned, so
    two signed companions are also reported:

``torso_lateral_lean_deg``
    Lean within the image plane: ``degrees(atan2(t_x, -t_y))``.
    Positive = leaning toward the right of the image.

``torso_forward_lean_deg``
    Lean along the camera axis: ``degrees(atan2(t_z, -t_y))``.
    Positive = leaning away from the camera.  This is the component a single
    camera estimates least reliably, and it should be read as indicative only.

``shoulder_tilt_deg`` / ``hip_tilt_deg``
    Tilt of the segment as it actually appears on screen, from pixel
    coordinates: ``wrap_to_90(degrees(atan2(dy, dx)))`` for the vector from the
    left to the right landmark.  Because image y points down, a positive value
    means the right side of the segment sits lower in the frame.  These are
    directly observable in the video, which makes them a good sanity check on
    the 3D numbers.

Interpretation limits
---------------------
Every number here is an estimate recovered from a single ordinary camera by a
learned model.  Rotation about the camera axis is the best conditioned;
rotation and lean along the camera axis (depth) are the worst.  None of this is
equivalent to a marker-based motion capture measurement.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..measurement import Measurement, build_measurement
from ..pose import landmarks as L

# Names of every rotation metric, with the landmarks each one depends on.
ROTATION_DEPENDENCIES: Dict[str, List[str]] = {
    "shoulder_orientation": ["left_shoulder", "right_shoulder"],
    "hip_orientation": ["left_hip", "right_hip"],
    "shoulder_hip_separation": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "torso_inclination": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "torso_lateral_lean": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "torso_forward_lean": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "shoulder_tilt": ["left_shoulder", "right_shoulder"],
    "hip_tilt": ["left_hip", "right_hip"],
}

ROTATION_SOURCES: Dict[str, str] = {
    "shoulder_orientation": "world_3d",
    "hip_orientation": "world_3d",
    "shoulder_hip_separation": "world_3d",
    "torso_inclination": "world_3d",
    "torso_lateral_lean": "world_3d",
    "torso_forward_lean": "world_3d",
    "shoulder_tilt": "image_2d",
    "hip_tilt": "image_2d",
}

# Which metrics wrap, and over what period. Statistics for these must be
# circular: torso_inclination is an unsigned magnitude in [0, 180] and does not
# wrap, so it is deliberately absent and gets ordinary linear statistics.
ROTATION_PERIODS: Dict[str, float] = {
    "shoulder_orientation": 360.0,
    "hip_orientation": 360.0,
    "shoulder_hip_separation": 360.0,
    "torso_lateral_lean": 360.0,
    "torso_forward_lean": 360.0,
    "shoulder_tilt": 180.0,
    "hip_tilt": 180.0,
}

ROTATION_DESCRIPTIONS: Dict[str, str] = {
    "shoulder_orientation": "Directed rotation of the shoulder line about the vertical axis "
                            "relative to the camera image plane, in (-180, 180] deg. 0 = player "
                            "square to camera facing away, +/-180 = square facing the camera, "
                            "-90 = side-on with the right shoulder nearer the camera.",
    "hip_orientation": "Directed rotation of the hip line about the vertical axis relative to "
                       "the camera image plane, in (-180, 180] deg. Same convention as "
                       "shoulder_orientation.",
    "shoulder_hip_separation": "Shoulder line orientation minus hip line orientation, wrapped to "
                               "(-180, 180] deg. The upper-to-lower body separation (X-factor).",
    "torso_inclination": "Unsigned angle between the trunk vector (hip midpoint to shoulder "
                         "midpoint) and true vertical, in [0, 180] deg. 0 = upright.",
    "torso_lateral_lean": "Signed trunk lean within the image plane, deg. "
                          "Positive = leaning toward the right of the image.",
    "torso_forward_lean": "Signed trunk lean along the camera axis, deg. Positive = leaning away "
                          "from the camera. Least reliable component on a single camera.",
    "shoulder_tilt": "On-screen tilt of the shoulder line from pixel coordinates, [-90, 90] deg. "
                     "Positive = right shoulder lower in the frame.",
    "hip_tilt": "On-screen tilt of the hip line from pixel coordinates, [-90, 90] deg. "
                "Positive = right hip lower in the frame.",
}

_MIN_SEGMENT_LENGTH = 1e-9


def wrap_to_180(degrees: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle into (-180, 180]."""
    return (np.asarray(degrees, dtype=float) + 180.0) % 360.0 - 180.0


def wrap_to_90(degrees: np.ndarray | float) -> np.ndarray | float:
    """
    Wrap an axis angle into (-90, 90].

    A body segment is an undirected line: rotating it by 180 deg gives the same
    line, so orientations must be compared modulo a half turn.
    """
    return (np.asarray(degrees, dtype=float) + 90.0) % 180.0 - 90.0


def unwrap_degrees(series: np.ndarray) -> np.ndarray:
    """
    Remove 360 degree jumps from an angle time series, ignoring gaps.

    ``np.unwrap`` cannot cope with NaN, so each contiguous run of valid samples
    is unwrapped independently.  Runs are not stitched to one another: bridging
    across an unobserved gap would be inventing rotation that was never seen.
    """
    out = np.asarray(series, dtype=float).copy()
    valid = np.isfinite(out)
    if not valid.any():
        return out
    start = None
    for i in range(len(out) + 1):
        is_valid = valid[i] if i < len(out) else False
        if is_valid and start is None:
            start = i
        elif not is_valid and start is not None:
            if i - start > 1:
                out[start:i] = np.degrees(np.unwrap(np.radians(out[start:i])))
            start = None
    return out


def segment_yaw_series(world_xyz: np.ndarray, left_name: str, right_name: str) -> np.ndarray:
    """
    Directed rotation of a left-right body segment about the vertical axis.

    Returned in (-180, 180]; see the module docstring for the sign convention.
    """
    left = L.point_from_array(world_xyz, left_name)
    right = L.point_from_array(world_xyz, right_name)
    v = right - left
    horizontal_length = np.hypot(v[..., 0], v[..., 2])
    with np.errstate(invalid="ignore"):
        yaw = np.degrees(np.arctan2(v[..., 2], v[..., 0]))
    return np.where(horizontal_length < _MIN_SEGMENT_LENGTH, np.nan, yaw)


def segment_tilt_series(image_xy: np.ndarray, left_name: str, right_name: str) -> np.ndarray:
    """On-screen tilt of a left-right body segment, in [-90, 90]."""
    left = L.point_from_array(image_xy, left_name)
    right = L.point_from_array(image_xy, right_name)
    v = right - left
    length = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore"):
        tilt = wrap_to_90(np.degrees(np.arctan2(v[..., 1], v[..., 0])))
    return np.where(length < _MIN_SEGMENT_LENGTH, np.nan, tilt)


def torso_vector_series(world_xyz: np.ndarray) -> np.ndarray:
    """Trunk vector: hip midpoint -> shoulder midpoint, in world coordinates."""
    shoulder_center = L.point_from_array(world_xyz, "shoulder_center")
    hip_center = L.point_from_array(world_xyz, "hip_center")
    return shoulder_center - hip_center


def torso_inclination_series(world_xyz: np.ndarray) -> np.ndarray:
    """Unsigned angle of the trunk from vertical, in degrees. 0 = upright."""
    t = torso_vector_series(world_xyz)
    length = np.linalg.norm(t, axis=-1)
    up = np.array([0.0, -1.0, 0.0])
    with np.errstate(invalid="ignore"):
        cross = np.linalg.norm(np.cross(t, up), axis=-1)
        dot = np.einsum("...i,i->...", t, up)
        inclination = np.degrees(np.arctan2(cross, dot))
    return np.where(length < _MIN_SEGMENT_LENGTH, np.nan, inclination)


def torso_lateral_lean_series(world_xyz: np.ndarray) -> np.ndarray:
    """Signed trunk lean in the image plane. Positive = toward image right."""
    t = torso_vector_series(world_xyz)
    length = np.linalg.norm(t, axis=-1)
    with np.errstate(invalid="ignore"):
        lean = np.degrees(np.arctan2(t[..., 0], -t[..., 1]))
    return np.where(length < _MIN_SEGMENT_LENGTH, np.nan, lean)


def torso_forward_lean_series(world_xyz: np.ndarray) -> np.ndarray:
    """Signed trunk lean along the camera axis. Positive = away from camera."""
    t = torso_vector_series(world_xyz)
    length = np.linalg.norm(t, axis=-1)
    with np.errstate(invalid="ignore"):
        lean = np.degrees(np.arctan2(t[..., 2], -t[..., 1]))
    return np.where(length < _MIN_SEGMENT_LENGTH, np.nan, lean)


def compute_rotations(sequence, config) -> Dict[str, List[Measurement]]:
    """
    Compute every body-orientation metric for every frame.

    Returns ``{metric_name: [Measurement per frame]}``.  Confidence for each
    metric is the minimum visibility across the landmarks it needs.
    """
    world = sequence.world_xyz
    image = sequence.image_xy

    raw_series: Dict[str, np.ndarray] = {
        "shoulder_orientation": segment_yaw_series(world, "left_shoulder", "right_shoulder"),
        "hip_orientation": segment_yaw_series(world, "left_hip", "right_hip"),
        "torso_inclination": torso_inclination_series(world),
        "torso_lateral_lean": torso_lateral_lean_series(world),
        "torso_forward_lean": torso_forward_lean_series(world),
        "shoulder_tilt": segment_tilt_series(image, "left_shoulder", "right_shoulder"),
        "hip_tilt": segment_tilt_series(image, "left_hip", "right_hip"),
    }
    raw_series["shoulder_hip_separation"] = wrap_to_180(
        raw_series["shoulder_orientation"] - raw_series["hip_orientation"]
    )

    results: Dict[str, List[Measurement]] = {}
    n_frames = len(sequence)
    for name, values in raw_series.items():
        confidences = L.combined_confidence(sequence.visibility, ROTATION_DEPENDENCIES[name])
        source = ROTATION_SOURCES[name]
        per_frame: List[Measurement] = []
        for i in range(n_frames):
            if not sequence.detected[i]:
                per_frame.append(
                    Measurement.unavailable(unit="deg", source=source, note="no pose detected")
                )
                continue
            per_frame.append(
                build_measurement(
                    value=float(values[i]),
                    confidence=float(confidences[i]),
                    confidence_threshold=config.confidence_threshold,
                    low_confidence_floor=config.low_confidence_floor,
                    unit="deg",
                    source=source,
                )
            )
        results[name] = per_frame
    return results
