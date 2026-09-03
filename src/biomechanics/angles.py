"""
Joint angles.

Definition
----------
For three points A, B, C with B as the vertex, the joint angle is the angle
between the vectors BA and BC::

        BA = A - B
        BC = C - B
        theta = atan2( |BA x BC| , BA . BC )

``atan2`` of the cross-product magnitude against the dot product is used rather
than ``arccos(dot / (|BA||BC|))`` because it stays numerically stable near 0 deg
and 180 deg, where the arccos form loses precision and can produce NaN from
floating point values slipping just outside [-1, 1].  Both forms give the same
answer; this one does not fall over at full extension, which is exactly where a
tennis elbow angle spends much of its time.

The result is in [0, 180] degrees and is unsigned: it is the interior angle at
the joint.  180 deg means fully extended (the three points are collinear), and
smaller values mean more flexed.

Which coordinates
-----------------
Angles are computed from the MediaPipe **world** landmarks by default: an
approximate metric 3D reconstruction whose axes are not distorted by the frame
aspect ratio.  A 2D image-plane angle is computed alongside it from pixel
coordinates, for comparison and for the on-screen overlay.

The 2D angle is a projection of the true joint angle onto the image plane, so
it is systematically wrong whenever the limb has a component along the camera
axis: a fully extended arm pointing at the camera projects to a sharply bent
angle.  Both numbers are exported and clearly labelled; the 3D one is the one
to trust, and even it is a single-camera estimate, not a laboratory measurement.

Note on pixel coordinates: computing an angle from the *normalised* [0, 1]
coordinates would be wrong on any non-square frame, because x and y are scaled
by different denominators, which shears the geometry.  Pixel coordinates are
used instead, which removes that error.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..measurement import Measurement, build_measurement
from ..pose import landmarks as L

# ---------------------------------------------------------------------------
# The joint angles required by the brief: name -> (point A, vertex B, point C)
# ---------------------------------------------------------------------------
JOINT_ANGLE_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
}

# Plain-language description of every angle, reused by the JSON report and README.
JOINT_ANGLE_DESCRIPTIONS: Dict[str, str] = {
    "right_elbow": "Interior angle at the right elbow, vertex right_elbow, "
                   "between the upper arm (to right_shoulder) and forearm (to right_wrist). "
                   "180 deg = fully extended.",
    "left_elbow": "Interior angle at the left elbow, vertex left_elbow, "
                  "between the upper arm (to left_shoulder) and forearm (to left_wrist). "
                  "180 deg = fully extended.",
    "right_knee": "Interior angle at the right knee, vertex right_knee, "
                  "between the thigh (to right_hip) and shank (to right_ankle). "
                  "180 deg = fully extended leg.",
    "left_knee": "Interior angle at the left knee, vertex left_knee, "
                 "between the thigh (to left_hip) and shank (to left_ankle). "
                 "180 deg = fully extended leg.",
    "right_hip": "Interior angle at the right hip, vertex right_hip, between the "
                 "trunk (to right_shoulder) and thigh (to right_knee). Smaller = more flexed.",
    "left_hip": "Interior angle at the left hip, vertex left_hip, between the "
                "trunk (to left_shoulder) and thigh (to left_knee). Smaller = more flexed.",
}

# Angles drawn on the video by default (drawing all six makes 500 px unreadable).
OVERLAY_ANGLES: List[str] = ["right_elbow", "left_elbow", "right_knee", "left_knee"]

# Below this vector length the geometry is degenerate and the angle meaningless.
_MIN_SEGMENT_LENGTH = 1e-6


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Interior angle at vertex ``b``, in degrees, in [0, 180].

    Works for 2D or 3D points.  Returns NaN when any coordinate is missing or
    when either segment is degenerately short, so that a missing input can
    never silently become a plausible-looking number.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        return float("nan")

    ba = a - b
    bc = c - b
    n_ba = float(np.linalg.norm(ba))
    n_bc = float(np.linalg.norm(bc))
    if n_ba < _MIN_SEGMENT_LENGTH or n_bc < _MIN_SEGMENT_LENGTH:
        return float("nan")

    if ba.shape[-1] == 2:
        cross = abs(float(ba[0] * bc[1] - ba[1] * bc[0]))
    else:
        cross = float(np.linalg.norm(np.cross(ba, bc)))
    dot = float(np.dot(ba, bc))
    return float(np.degrees(np.arctan2(cross, dot)))


def angle_series(coords: np.ndarray, a_name: str, b_name: str, c_name: str) -> np.ndarray:
    """Vectorised joint angle over a whole clip. ``coords`` is (T, 33, D)."""
    a = L.point_from_array(coords, a_name)
    b = L.point_from_array(coords, b_name)
    c = L.point_from_array(coords, c_name)

    ba = a - b
    bc = c - b
    n_ba = np.linalg.norm(ba, axis=-1)
    n_bc = np.linalg.norm(bc, axis=-1)

    if ba.shape[-1] == 2:
        cross = np.abs(ba[..., 0] * bc[..., 1] - ba[..., 1] * bc[..., 0])
    else:
        cross = np.linalg.norm(np.cross(ba, bc), axis=-1)
    dot = np.einsum("...i,...i->...", ba, bc)

    with np.errstate(invalid="ignore"):
        out = np.degrees(np.arctan2(cross, dot))
    degenerate = (n_ba < _MIN_SEGMENT_LENGTH) | (n_bc < _MIN_SEGMENT_LENGTH)
    out = np.where(degenerate, np.nan, out)
    return out


def compute_joint_angles(sequence, config) -> Dict[str, List[Measurement]]:
    """
    Compute every defined joint angle for every frame of a sequence.

    Returns ``{angle_name: [Measurement per frame]}`` using the 3D world
    landmarks.  Confidence for an angle is the minimum visibility among its
    three contributing landmarks, so an occluded wrist immediately downgrades
    the elbow angle that depends on it.
    """
    results: Dict[str, List[Measurement]] = {}
    n_frames = len(sequence)

    for name, (a, b, c) in JOINT_ANGLE_DEFINITIONS.items():
        values = angle_series(sequence.world_xyz, a, b, c)
        confidences = L.combined_confidence(sequence.visibility, [a, b, c])
        per_frame: List[Measurement] = []
        for i in range(n_frames):
            if not sequence.detected[i]:
                per_frame.append(
                    Measurement.unavailable(unit="deg", source="world_3d", note="no pose detected")
                )
                continue
            per_frame.append(
                build_measurement(
                    value=float(values[i]),
                    confidence=float(confidences[i]),
                    confidence_threshold=config.confidence_threshold,
                    low_confidence_floor=config.low_confidence_floor,
                    unit="deg",
                    source="world_3d",
                )
            )
        results[name] = per_frame
    return results


def compute_joint_angles_2d(sequence, config) -> Dict[str, List[Measurement]]:
    """
    The same angles measured in the image plane, from pixel coordinates.

    Kept separate and clearly labelled ``image_2d``.  These are what a viewer
    sees on screen, which makes them useful for the overlay and for sanity
    checking, but they are projections and will disagree with the 3D values
    whenever the limb is not parallel to the image plane.
    """
    results: Dict[str, List[Measurement]] = {}
    n_frames = len(sequence)

    for name, (a, b, c) in JOINT_ANGLE_DEFINITIONS.items():
        values = angle_series(sequence.image_xy, a, b, c)
        confidences = L.combined_confidence(sequence.visibility, [a, b, c])
        per_frame: List[Measurement] = []
        for i in range(n_frames):
            if not sequence.detected[i]:
                per_frame.append(
                    Measurement.unavailable(unit="deg", source="image_2d", note="no pose detected")
                )
                continue
            per_frame.append(
                build_measurement(
                    value=float(values[i]),
                    confidence=float(confidences[i]),
                    confidence_threshold=config.confidence_threshold,
                    low_confidence_floor=config.low_confidence_floor,
                    unit="deg",
                    source="image_2d",
                )
            )
        results[name] = per_frame
    return results
