"""
Central configuration for the analysis pipeline.

The module-level constants below are the "quick toggles" described in the POC
brief.  They are read once at import time to build the default
:class:`AnalysisConfig`; every one of them can also be overridden per-run from
the command line (see ``analyze.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Visualisation toggles
# ---------------------------------------------------------------------------
SHOW_SKELETON = True
SHOW_ANGLES = True
SHOW_WRIST_TRAIL = True
SHOW_HIP_TRAIL = True
SHOW_FOOT_TRAILS = True
SHOW_HUD = True             # frame number / timestamp / confidence panel
SHOW_ORIENTATION = True     # shoulder & hip lines drawn on the torso

# ---------------------------------------------------------------------------
# Confidence policy
# ---------------------------------------------------------------------------
# A metric is only reported as MEASURED when every landmark it depends on has a
# visibility score >= CONFIDENCE_THRESHOLD.  Between LOW_CONFIDENCE_FLOOR and
# CONFIDENCE_THRESHOLD the value is kept but flagged LOW_CONFIDENCE (and is
# excluded from the summary statistics).  Below LOW_CONFIDENCE_FLOOR nothing is
# computed at all: the metric is UNAVAILABLE and no number is invented.
CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_FLOOR = 0.30

# Minimum detector confidences handed to MediaPipe.
MIN_POSE_DETECTION_CONFIDENCE = 0.50
MIN_POSE_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50

# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------
SMOOTHING_METHOD = "savgol"     # "savgol" | "oneeuro" | "none"
SAVGOL_WINDOW = 9               # frames (forced odd, auto-shrunk on short clips)
SAVGOL_POLYORDER = 2
# Longest run of consecutive missing frames that will be bridged by linear
# interpolation before smoothing.  Longer gaps stay NaN -> UNAVAILABLE.
MAX_INTERPOLATION_GAP = 5

# One Euro filter parameters (only used when SMOOTHING_METHOD == "oneeuro").
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.03
ONE_EURO_D_CUTOFF = 1.0

# Pose models occasionally throw a landmark tens of pixels away for a single
# frame while still reporting high visibility. Confidence filtering cannot see
# that, so a geometric there-and-back test removes those samples before
# smoothing. Affected landmarks are capped below the MEASURED threshold.
SPIKE_REJECTION = True
SPIKE_FACTOR = 4.0        # multiples of the median frame-to-frame step

# ---------------------------------------------------------------------------
# Trails
# ---------------------------------------------------------------------------
TRAIL_LENGTH = 30               # frames of history drawn behind the joint
TRAIL_THICKNESS = 3

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
ANNOTATED_VIDEO_NAME = "annotated_forehand.mp4"
FRAME_CSV_NAME = "frame_metrics.csv"
ANALYSIS_JSON_NAME = "analysis.json"

# Fallback FPS used only when the container reports nothing usable.
FALLBACK_FPS = 30.0


@dataclass
class AnalysisConfig:
    """Everything one run of the pipeline needs to know."""

    # --- I/O ---------------------------------------------------------------
    video_path: Path = Path("input/forehand.mp4")
    output_dir: Path = Path(OUTPUT_DIR)
    annotated_video_name: str = ANNOTATED_VIDEO_NAME
    frame_csv_name: str = FRAME_CSV_NAME
    analysis_json_name: str = ANALYSIS_JSON_NAME

    # --- Pose backend ------------------------------------------------------
    backend: str = "mediapipe"
    model_complexity: str = "full"      # "lite" | "full" | "heavy"
    min_detection_confidence: float = MIN_POSE_DETECTION_CONFIDENCE
    min_presence_confidence: float = MIN_POSE_PRESENCE_CONFIDENCE
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE

    # --- Confidence policy -------------------------------------------------
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    low_confidence_floor: float = LOW_CONFIDENCE_FLOOR

    # --- Smoothing ---------------------------------------------------------
    smoothing_method: str = SMOOTHING_METHOD
    savgol_window: int = SAVGOL_WINDOW
    savgol_polyorder: int = SAVGOL_POLYORDER
    max_interpolation_gap: int = MAX_INTERPOLATION_GAP
    one_euro_min_cutoff: float = ONE_EURO_MIN_CUTOFF
    one_euro_beta: float = ONE_EURO_BETA
    one_euro_d_cutoff: float = ONE_EURO_D_CUTOFF
    spike_rejection: bool = SPIKE_REJECTION
    spike_factor: float = SPIKE_FACTOR

    # --- Player ------------------------------------------------------------
    # "right" | "left" | None.  None keeps both wrists on equal footing.
    hand: Optional[str] = None
    # Optional anthropometric prior used ONLY to express image-space distances
    # in approximate metres.  Always reported as an estimate.
    player_height_m: Optional[float] = None

    # --- Visualisation -----------------------------------------------------
    show_skeleton: bool = SHOW_SKELETON
    show_angles: bool = SHOW_ANGLES
    show_wrist_trail: bool = SHOW_WRIST_TRAIL
    show_hip_trail: bool = SHOW_HIP_TRAIL
    show_foot_trails: bool = SHOW_FOOT_TRAILS
    show_hud: bool = SHOW_HUD
    show_orientation: bool = SHOW_ORIENTATION
    trail_length: int = TRAIL_LENGTH
    trail_thickness: int = TRAIL_THICKNESS
    draw_source: str = "smoothed"       # "smoothed" | "raw"
    # Render the annotated video larger than the source so a small clip stays
    # readable.  None = automatic (upscale toward 720 px wide, never above 2x).
    output_scale: Optional[float] = None

    # --- Behaviour ---------------------------------------------------------
    write_video: bool = True
    write_plots: bool = True
    fallback_fps: float = FALLBACK_FPS
    max_frames: Optional[int] = None    # debugging aid

    # ------------------------------------------------------------------
    @property
    def annotated_video_path(self) -> Path:
        return self.output_dir / self.annotated_video_name

    @property
    def frame_csv_path(self) -> Path:
        return self.output_dir / self.frame_csv_name

    @property
    def analysis_json_path(self) -> Path:
        return self.output_dir / self.analysis_json_name

    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    def validate(self) -> None:
        if not 0.0 <= self.low_confidence_floor <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "Confidence settings must satisfy "
                "0 <= low_confidence_floor <= confidence_threshold <= 1 "
                f"(got floor={self.low_confidence_floor}, "
                f"threshold={self.confidence_threshold})"
            )
        if self.hand not in (None, "left", "right"):
            raise ValueError(f"hand must be 'left', 'right' or None (got {self.hand!r})")
        if self.smoothing_method not in ("savgol", "oneeuro", "none"):
            raise ValueError(
                f"Unknown smoothing method {self.smoothing_method!r}; "
                "expected 'savgol', 'oneeuro' or 'none'"
            )
        if self.player_height_m is not None and not (0.5 < self.player_height_m < 2.6):
            raise ValueError(
                f"player_height_m={self.player_height_m} is outside the plausible "
                "range 0.5-2.6 m"
            )
        if self.savgol_polyorder >= self.savgol_window:
            raise ValueError(
                f"savgol_window ({self.savgol_window}) must be greater than "
                f"savgol_polyorder ({self.savgol_polyorder})"
            )
        if self.spike_factor <= 1.0:
            raise ValueError(
                f"spike_factor must be greater than 1 (got {self.spike_factor}); "
                "smaller values would reject ordinary movement as outliers"
            )
        if self.output_scale is not None and not (0.1 <= self.output_scale <= 4.0):
            raise ValueError(f"output_scale must be between 0.1 and 4.0 (got {self.output_scale})")
        if self.model_complexity not in ("lite", "full", "heavy"):
            raise ValueError(
                f"model_complexity must be 'lite', 'full' or 'heavy' "
                f"(got {self.model_complexity!r})"
            )

    def resolve_output_scale(self, width: int) -> float:
        """
        Pick the render scale for the annotated video.

        Small clips get upscaled so the overlay text and skeleton stay legible;
        anything already reasonably wide is left at native resolution.  The
        source pixels are never treated as more precise than they are: this is
        a presentation choice only, and all measurements stay in source pixels.
        """
        if self.output_scale is not None:
            return float(self.output_scale)
        if width <= 0:
            return 1.0
        return float(min(2.0, max(1.0, 720.0 / width)))
