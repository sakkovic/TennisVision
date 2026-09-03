"""
Scale normalisation.

A monocular camera gives no absolute scale.  Two clips of the same stroke, one
filmed from ten metres and one from twenty, produce pixel displacements that
differ by a factor of two while the movement is identical.  Anything reported
purely in pixels is therefore comparable only within a single clip.

This module provides two ways out, in increasing order of assumption:

1. **Body units** (no extra assumptions).
   Distances are divided by the player's own torso length in pixels, giving
   "torso lengths".  This cancels camera distance and image resolution and is
   the safest normalisation available from the video alone.  The reference
   torso length is the median over confidently measured frames, which resists
   both jitter and the occasional bad frame.

2. **Approximate metres** (requires the player height).
   Only when ``--player-height`` is supplied.  Standard anthropometry puts the
   shoulder joint at about 0.818 of stature and the hip joint at about 0.530
   (Winter, *Biomechanics and Motor Control of Human Movement*), so the trunk
   segment measured here spans roughly ``0.288 x height``.  Combining that with
   the measured torso length in pixels yields a metres-per-pixel scale.

   This is an estimate with real error sources: the ratio varies between
   individuals, and it assumes the player is roughly side-on to the camera and
   near the optical axis, so a torso pointing toward the camera is foreshortened
   and the scale is overestimated.  Values derived this way are always labelled
   as estimates and are never presented as measured distances.

Perspective is not corrected anywhere.  A player moving toward the camera grows
in the frame, and no homography or court-line calibration is applied in this
phase, so displacements measured in different parts of the court are not
strictly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..pose import landmarks as L

# Winter's anthropometric ratios: fraction of stature at each joint centre.
SHOULDER_HEIGHT_RATIO = 0.818
HIP_HEIGHT_RATIO = 0.530
TRUNK_SEGMENT_RATIO = SHOULDER_HEIGHT_RATIO - HIP_HEIGHT_RATIO   # ~0.288


@dataclass
class BodyScale:
    """Reference scales for one clip, derived from the player themselves."""

    torso_length_px: Optional[float]
    shoulder_width_px: Optional[float]
    frames_used: int
    metres_per_pixel: Optional[float] = None
    player_height_m: Optional[float] = None

    @property
    def has_body_units(self) -> bool:
        return self.torso_length_px is not None and self.torso_length_px > 1e-6

    @property
    def has_metric_estimate(self) -> bool:
        return self.metres_per_pixel is not None and self.metres_per_pixel > 0

    # -- conversions ----------------------------------------------------
    def px_to_body_units(self, value_px: float | np.ndarray):
        """Pixels -> torso lengths.  Returns NaN when no reference exists."""
        if not self.has_body_units:
            return np.full_like(np.asarray(value_px, dtype=float), np.nan)
        return np.asarray(value_px, dtype=float) / self.torso_length_px

    def px_to_metres(self, value_px: float | np.ndarray):
        """Pixels -> approximate metres.  Returns NaN when no height was given."""
        if not self.has_metric_estimate:
            return np.full_like(np.asarray(value_px, dtype=float), np.nan)
        return np.asarray(value_px, dtype=float) * self.metres_per_pixel

    def to_dict(self) -> Dict:
        return {
            "torso_length_px": round(self.torso_length_px, 2) if self.torso_length_px else None,
            "shoulder_width_px": round(self.shoulder_width_px, 2)
            if self.shoulder_width_px else None,
            "frames_used_for_reference": self.frames_used,
            "player_height_m": self.player_height_m,
            "metres_per_pixel_estimate": round(self.metres_per_pixel, 6)
            if self.metres_per_pixel else None,
            "method": (
                "Torso length = median pixel distance from the hip midpoint to the shoulder "
                "midpoint over frames where all four torso landmarks were MEASURED. "
                "Body units express distances as multiples of that length."
            ),
            "metric_method": (
                "metres_per_pixel = (0.288 x player_height_m) / torso_length_px, using Winter "
                "anthropometric ratios (shoulder 0.818H, hip 0.530H). ESTIMATE ONLY: assumes the "
                "trunk is roughly perpendicular to the camera axis and ignores perspective."
                if self.has_metric_estimate else
                "Not computed: no --player-height was supplied, so no metric scale is claimed."
            ),
        }


def _segment_length_px(image_xy: np.ndarray, a_name: str, b_name: str) -> np.ndarray:
    a = L.point_from_array(image_xy, a_name)
    b = L.point_from_array(image_xy, b_name)
    return np.linalg.norm(a - b, axis=-1)


def compute_body_scale(sequence, config) -> BodyScale:
    """
    Derive the clip reference scale from confidently measured torso frames.

    Only frames where all four torso landmarks reach ``confidence_threshold``
    contribute.  If no frame qualifies, the scale is left undefined rather than
    guessed, and everything downstream that needs it reports UNAVAILABLE.
    """
    torso_conf = L.combined_confidence(sequence.visibility, L.TORSO_LANDMARKS)
    usable = sequence.detected & (torso_conf >= config.confidence_threshold)

    torso_px = _segment_length_px(sequence.image_xy, "shoulder_center", "hip_center")
    shoulder_px = _segment_length_px(sequence.image_xy, "left_shoulder", "right_shoulder")

    torso_valid = torso_px[usable & np.isfinite(torso_px)]
    shoulder_valid = shoulder_px[usable & np.isfinite(shoulder_px)]

    torso_ref = float(np.median(torso_valid)) if torso_valid.size else None
    shoulder_ref = float(np.median(shoulder_valid)) if shoulder_valid.size else None

    metres_per_pixel = None
    height = getattr(config, "player_height_m", None)
    if height and torso_ref and torso_ref > 1e-6:
        metres_per_pixel = (TRUNK_SEGMENT_RATIO * float(height)) / torso_ref

    return BodyScale(
        torso_length_px=torso_ref,
        shoulder_width_px=shoulder_ref,
        frames_used=int(torso_valid.size),
        metres_per_pixel=metres_per_pixel,
        player_height_m=float(height) if height else None,
    )


def normalized_to_pixels(normalized_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Image-normalised [0, 1] coordinates -> pixels."""
    out = np.asarray(normalized_xy, dtype=float).copy()
    out[..., 0] *= width
    out[..., 1] *= height
    return out


def pixels_to_normalized(pixel_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixels -> image-normalised [0, 1] coordinates."""
    out = np.asarray(pixel_xy, dtype=float).copy()
    out[..., 0] /= max(width, 1)
    out[..., 1] /= max(height, 1)
    return out
