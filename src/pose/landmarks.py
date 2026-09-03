"""
Landmark vocabulary shared by every module.

The indices follow the 33-point BlazePose / MediaPipe Pose topology.  Keeping
this mapping in one place is what allows the detector backend to be swapped
later (YOLO-Pose, RTMPose, a PyTorch model) without touching the biomechanics:
a new backend only has to emit arrays in this landmark order, or supply its own
index mapping onto these names.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# The 33 landmarks of the MediaPipe Pose topology, in index order.
# ---------------------------------------------------------------------------
LANDMARK_NAMES: List[str] = [
    "nose",                 # 0
    "left_eye_inner",       # 1
    "left_eye",             # 2
    "left_eye_outer",       # 3
    "right_eye_inner",      # 4
    "right_eye",            # 5
    "right_eye_outer",      # 6
    "left_ear",             # 7
    "right_ear",            # 8
    "mouth_left",           # 9
    "mouth_right",          # 10
    "left_shoulder",        # 11
    "right_shoulder",       # 12
    "left_elbow",           # 13
    "right_elbow",          # 14
    "left_wrist",           # 15
    "right_wrist",          # 16
    "left_pinky",           # 17
    "right_pinky",          # 18
    "left_index",           # 19
    "right_index",          # 20
    "left_thumb",           # 21
    "right_thumb",          # 22
    "left_hip",             # 23
    "right_hip",            # 24
    "left_knee",            # 25
    "right_knee",           # 26
    "left_ankle",           # 27
    "right_ankle",          # 28
    "left_heel",            # 29
    "right_heel",           # 30
    "left_foot_index",      # 31
    "right_foot_index",     # 32
]

NUM_LANDMARKS = len(LANDMARK_NAMES)

IDX: Dict[str, int] = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# ---------------------------------------------------------------------------
# The subset the brief requires us to track and export.  Everything else is
# still detected and available, it just does not clutter the CSV.
# ---------------------------------------------------------------------------
TRACKED_LANDMARKS: List[str] = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Landmarks that define the torso; used for scale normalisation and rotation.
TORSO_LANDMARKS: List[str] = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
]

# ---------------------------------------------------------------------------
# Skeleton edges used for drawing.  Grouped so the overlay can colour body
# regions differently, which makes a 500 px wide frame far easier to read.
# ---------------------------------------------------------------------------
SKELETON_GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "torso": [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ],
    "left_arm": [
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
    ],
    "right_arm": [
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ],
    "left_leg": [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("left_ankle", "left_heel"),
        ("left_heel", "left_foot_index"),
        ("left_ankle", "left_foot_index"),
    ],
    "right_leg": [
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("right_ankle", "right_heel"),
        ("right_heel", "right_foot_index"),
        ("right_ankle", "right_foot_index"),
    ],
    "head": [
        ("left_ear", "left_eye"),
        ("left_eye", "nose"),
        ("nose", "right_eye"),
        ("right_eye", "right_ear"),
    ],
}

SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    edge for edges in SKELETON_GROUPS.values() for edge in edges
]

# Joints drawn as filled circles.
JOINT_LANDMARKS: List[str] = TRACKED_LANDMARKS + ["nose"]


# ---------------------------------------------------------------------------
# Derived (virtual) points.  A derived point is the mean of a set of real
# landmarks; its confidence is the *minimum* confidence of its parts, so a
# midpoint is never more trusted than its weakest ingredient.
# ---------------------------------------------------------------------------
DERIVED_POINTS: Dict[str, Tuple[str, ...]] = {
    # Midpoint between the two hips.  Used as the body reference point.
    # NOTE: this is a geometric midpoint of two surface landmarks. It is a
    # proxy for pelvis position, NOT the body centre of mass.
    "hip_center": ("left_hip", "right_hip"),
    # Midpoint between the two shoulders (proxy for the base of the neck).
    "shoulder_center": ("left_shoulder", "right_shoulder"),
    # Foot reference points: midpoint of heel and toe, which is steadier than
    # either landmark alone when the foot rolls through a step.
    "left_foot": ("left_heel", "left_foot_index"),
    "right_foot": ("right_heel", "right_foot_index"),
}

# Every point name the rest of the pipeline may ask for.
ALL_POINT_NAMES: List[str] = LANDMARK_NAMES + list(DERIVED_POINTS.keys())


def is_derived(name: str) -> bool:
    return name in DERIVED_POINTS


def resolve_indices(name: str) -> Tuple[int, ...]:
    """Return the landmark indices a point name depends on."""
    if name in DERIVED_POINTS:
        return tuple(IDX[part] for part in DERIVED_POINTS[name])
    if name in IDX:
        return (IDX[name],)
    raise KeyError(f"Unknown landmark or derived point: {name!r}")


def point_from_array(coords: np.ndarray, name: str) -> np.ndarray:
    """
    Extract one point from a landmark array.

    ``coords`` is ``(..., NUM_LANDMARKS, D)``.  Derived points are averaged
    across their constituent landmarks.  NaNs propagate, which is exactly what
    we want: a midpoint of a missing landmark is itself missing.
    """
    indices = resolve_indices(name)
    if len(indices) == 1:
        return coords[..., indices[0], :]
    return np.mean(coords[..., indices, :], axis=-2)


def confidence_from_array(visibility: np.ndarray, name: str) -> np.ndarray:
    """
    Confidence of a point: the minimum visibility over its constituents.

    Using the minimum (not the mean) is deliberate. A midpoint built from one
    clearly visible landmark and one occluded landmark is an unreliable
    midpoint, and the confidence must say so.
    """
    indices = resolve_indices(name)
    if len(indices) == 1:
        return visibility[..., indices[0]]
    return np.min(visibility[..., indices], axis=-1)


def combined_confidence(visibility: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Minimum confidence across several points (used for multi-point metrics)."""
    stacked = np.stack([confidence_from_array(visibility, n) for n in names], axis=-1)
    return np.min(stacked, axis=-1)
