"""Shared fixtures: synthetic pose sequences with known ground truth."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AnalysisConfig            # noqa: E402
from src.pose import landmarks as L              # noqa: E402
from src.pose.detector import PoseSequence       # noqa: E402


@pytest.fixture
def config() -> AnalysisConfig:
    return AnalysisConfig(
        confidence_threshold=0.5,
        low_confidence_floor=0.3,
        savgol_window=9,
        savgol_polyorder=2,
        max_interpolation_gap=5,
    )


def make_sequence(n_frames: int = 30, fps: float = 30.0, width: int = 640,
                  height: int = 480, visibility: float = 0.9) -> PoseSequence:
    """
    A synthetic standing figure with a known geometry.

    The body is built directly in world coordinates so every derived quantity
    has an exact expected value: the arms are bent to a right angle, the legs
    are straight, and the shoulder and hip lines are parallel to the image
    plane.
    """
    sequence = PoseSequence.empty(n_frames, width, height, fps)
    sequence.timestamps = np.arange(n_frames) / fps
    sequence.detected[:] = True
    sequence.visibility[:] = visibility
    sequence.presence[:] = visibility

    world = np.zeros((L.NUM_LANDMARKS, 3), dtype=float)
    # y is down, so negative y is above the hip origin.
    world[L.IDX["left_shoulder"]] = (-0.20, -0.50, 0.0)
    world[L.IDX["right_shoulder"]] = (0.20, -0.50, 0.0)
    world[L.IDX["left_hip"]] = (-0.12, 0.0, 0.0)
    world[L.IDX["right_hip"]] = (0.12, 0.0, 0.0)
    # Elbows straight below the shoulders, wrists straight out sideways:
    # upper arm points down, forearm points outward => 90 degrees at the elbow.
    world[L.IDX["left_elbow"]] = (-0.20, -0.20, 0.0)
    world[L.IDX["right_elbow"]] = (0.20, -0.20, 0.0)
    world[L.IDX["left_wrist"]] = (-0.50, -0.20, 0.0)
    world[L.IDX["right_wrist"]] = (0.50, -0.20, 0.0)
    # Straight legs => 180 degrees at the knee.
    world[L.IDX["left_knee"]] = (-0.12, 0.45, 0.0)
    world[L.IDX["right_knee"]] = (0.12, 0.45, 0.0)
    world[L.IDX["left_ankle"]] = (-0.12, 0.90, 0.0)
    world[L.IDX["right_ankle"]] = (0.12, 0.90, 0.0)
    world[L.IDX["left_heel"]] = (-0.12, 0.95, -0.03)
    world[L.IDX["right_heel"]] = (0.12, 0.95, -0.03)
    world[L.IDX["left_foot_index"]] = (-0.12, 0.95, 0.12)
    world[L.IDX["right_foot_index"]] = (0.12, 0.95, 0.12)
    world[L.IDX["nose"]] = (0.0, -0.70, 0.05)

    for i in range(n_frames):
        sequence.world_xyz[i] = world

    # Project into the image with a simple scale + centre offset, then derive
    # the normalised coordinates from the pixels so the two stay consistent.
    pixels_per_metre = 200.0
    sequence.image_xy[:, :, 0] = world[None, :, 0] * pixels_per_metre + width / 2.0
    sequence.image_xy[:, :, 1] = world[None, :, 1] * pixels_per_metre + height / 2.0
    sequence.normalized_xyz[:, :, 0] = sequence.image_xy[:, :, 0] / width
    sequence.normalized_xyz[:, :, 1] = sequence.image_xy[:, :, 1] / height
    sequence.normalized_xyz[:, :, 2] = world[None, :, 2]
    return sequence


@pytest.fixture
def sequence() -> PoseSequence:
    return make_sequence()


@pytest.fixture
def sample_video() -> Path:
    path = ROOT / "input" / "sample_serve.mp4"
    if not path.exists():
        pytest.skip("bundled sample clip is not present")
    return path
