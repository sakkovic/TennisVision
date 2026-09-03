"""
Result assembly: the per-frame CSV, the JSON summary and the plots.

Two conventions run through everything here:

* A quantity that was not measured is empty in the CSV and ``null`` in the
  JSON.  It is never zero, never carried over from the previous frame and never
  interpolated.  Every metric is accompanied by its status and the confidence
  it was derived from.
* Summary statistics are computed from ``MEASURED`` samples only, and each one
  is published together with how many frames it actually rests on, so a mean
  over six good frames cannot be mistaken for a mean over the whole clip.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..biomechanics.angles import JOINT_ANGLE_DEFINITIONS, JOINT_ANGLE_DESCRIPTIONS
from ..biomechanics.normalization import BodyScale
from ..biomechanics.rotations import (
    ROTATION_DESCRIPTIONS, ROTATION_PERIODS, unwrap_degrees,
)
from ..biomechanics.trajectories import Trajectory
from ..measurement import Measurement, MeasurementStatus, summarise, summarise_angular
from ..pose import landmarks as L

logger = logging.getLogger(__name__)

# Text reproduced in every report so the numbers are never read as more than
# they are.
LIMITATIONS: List[str] = [
    "All measurements are estimated from a single ordinary (monocular) video. They are not "
    "motion capture and carry no marker-based ground truth.",
    "3D landmark positions come from a learned statistical model of human body shape. Depth "
    "along the camera axis is the least reliable component, so any quantity that depends on it "
    "(forward lean, rotation toward or away from the camera) carries the largest error.",
    "hip_center is the geometric midpoint of the two hip landmarks. It is a proxy for pelvis "
    "position and is NOT the body centre of mass.",
    "Joint angles are the interior angles between adjacent body segments as reconstructed by the "
    "pose model. They are not clinical goniometry and must not be used for diagnosis.",
    "Image-plane (2D) angles are projections and will differ from the 3D values whenever a limb "
    "is not parallel to the image plane.",
    "Distances and speeds in pixels are only comparable within this one clip. No court "
    "calibration or perspective correction is applied, so a player closer to the camera "
    "registers larger pixel motion for the same real movement.",
    "Metre values, when present, are estimates derived from an assumed player height and "
    "standard anthropometric ratios, measured in the image plane only.",
    "No ball tracking is performed.",
    "No racket tracking is performed.",
    "No stroke classification is performed: the clip is analysed as continuous motion, and "
    "nothing in this report identifies a stroke type or phase.",
    "No racket-ball contact point is detected or inferred.",
    "No comparison against professional players is made.",
    "No technique quality score, grade or coaching judgement is produced. This phase reports "
    "objective measurements only; tennis-specific interpretation requires validated criteria "
    "that have not yet been defined.",
    "Handedness is taken from the --hand argument when supplied and is never inferred from the "
    "video.",
    "Only one person is analysed per clip. When several people are visible the largest, most "
    "central and most temporally consistent candidate is chosen by a simple heuristic.",
]


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def build_frame_dataframe(
    raw_sequence,
    smoothed_sequence,
    angles_3d: Dict[str, List[Measurement]],
    angles_2d: Dict[str, List[Measurement]],
    rotations: Dict[str, List[Measurement]],
    trajectories: Dict[str, Trajectory],
) -> pd.DataFrame:
    """
    One row per frame.

    Column naming
    -------------
    ``<landmark>_x`` / ``_y``       smoothed pixel coordinates (primary)
    ``<landmark>_x_raw`` / ``_y_raw``  unsmoothed pixel coordinates
    ``<landmark>_z``                smoothed relative depth, MediaPipe normalised
                                    units, hip midpoint as origin, negative is
                                    nearer the camera
    ``<landmark>_confidence``       landmark visibility in [0, 1]
    ``<metric>``                    the value, blank when not measured
    ``<metric>_status``             MEASURED / LOW_CONFIDENCE / UNAVAILABLE
    ``<metric>_confidence``         confidence the status was derived from
    """
    n = len(smoothed_sequence)
    confidence = smoothed_sequence.frame_confidence()

    data: Dict[str, object] = {
        "frame": smoothed_sequence.frame_indices,
        "timestamp": np.round(smoothed_sequence.timestamps, 4),
        "pose_detected": smoothed_sequence.detected,
        "pose_confidence": np.round(confidence, 4),
    }

    # --- landmarks ------------------------------------------------------
    for name in L.TRACKED_LANDMARKS:
        idx = L.IDX[name]
        smooth_xy = smoothed_sequence.image_xy[:, idx, :]
        raw_xy = raw_sequence.image_xy[:, idx, :]
        data[f"{name}_x"] = np.round(smooth_xy[:, 0], 3)
        data[f"{name}_y"] = np.round(smooth_xy[:, 1], 3)
        data[f"{name}_z"] = np.round(smoothed_sequence.normalized_xyz[:, idx, 2], 5)
        data[f"{name}_confidence"] = np.round(smoothed_sequence.visibility[:, idx], 4)
        data[f"{name}_x_raw"] = np.round(raw_xy[:, 0], 3)
        data[f"{name}_y_raw"] = np.round(raw_xy[:, 1], 3)

    # --- derived points --------------------------------------------------
    for name in ("hip_center", "shoulder_center", "left_foot", "right_foot"):
        smooth_point = L.point_from_array(smoothed_sequence.image_xy, name)
        raw_point = L.point_from_array(raw_sequence.image_xy, name)
        point_conf = L.confidence_from_array(smoothed_sequence.visibility, name)
        data[f"{name}_x"] = np.round(smooth_point[:, 0], 3)
        data[f"{name}_y"] = np.round(smooth_point[:, 1], 3)
        data[f"{name}_confidence"] = np.round(point_conf, 4)
        data[f"{name}_x_raw"] = np.round(raw_point[:, 0], 3)
        data[f"{name}_y_raw"] = np.round(raw_point[:, 1], 3)

    # --- joint angles ----------------------------------------------------
    def add_measurements(prefix: str, series: List[Measurement]) -> None:
        data[prefix] = [m.as_float() for m in series]
        data[f"{prefix}_status"] = [m.status.value for m in series]
        data[f"{prefix}_confidence"] = [round(m.confidence, 4) for m in series]

    for name in JOINT_ANGLE_DEFINITIONS:
        add_measurements(f"{name}_angle", angles_3d[name])
        data[f"{name}_angle_2d"] = [m.as_float() for m in angles_2d[name]]

    # --- body orientation -------------------------------------------------
    for name, series in rotations.items():
        add_measurements(name, series)

    # Continuous (unwrapped) versions of the two directed yaw angles, so a
    # player rotating through the +/-180 branch cut plots as a smooth curve.
    for name in ("shoulder_orientation", "hip_orientation"):
        values = np.array([m.as_float() for m in rotations[name]], dtype=float)
        data[f"{name}_unwrapped"] = np.round(unwrap_degrees(values), 3)

    # --- trajectory speeds -------------------------------------------------
    for name, trajectory in trajectories.items():
        speed = trajectory.speed_px_s
        data[f"{name}_speed_px_s"] = np.round(speed, 3) if speed.size else np.full(n, np.nan)

    frame = pd.DataFrame(data)

    # Blank out any value whose status says it was not measured, so a reader
    # skimming the numeric columns cannot pick up a number that the status
    # column disowns.
    for column in list(frame.columns):
        status_column = f"{column}_status"
        if status_column in frame.columns:
            frame.loc[frame[status_column] == MeasurementStatus.UNAVAILABLE.value, column] = np.nan
    return frame


def write_frame_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows x %d columns)", path, len(frame), len(frame.columns))
    return path


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _quality_block(sequence, config) -> Dict:
    confidence = sequence.frame_confidence()
    n = len(sequence)
    detected = sequence.detected

    measured = int(np.count_nonzero(detected & (confidence >= config.confidence_threshold)))
    low = int(np.count_nonzero(
        detected & (confidence >= config.low_confidence_floor)
        & (confidence < config.confidence_threshold)
    ))
    unusable = n - measured - low
    detected_confidence = confidence[detected]

    return {
        "frames_analysed": n,
        "frames_with_pose_detected": int(np.count_nonzero(detected)),
        "frames_without_pose": int(n - np.count_nonzero(detected)),
        "average_pose_confidence": round(float(np.mean(detected_confidence)), 4)
        if detected_confidence.size else None,
        "min_pose_confidence": round(float(np.min(detected_confidence)), 4)
        if detected_confidence.size else None,
        "max_pose_confidence": round(float(np.max(detected_confidence)), 4)
        if detected_confidence.size else None,
        "measured_frames": measured,
        "low_confidence_frames": low,
        "unusable_frames": int(unusable),
        "usable_frame_percentage": round(100.0 * measured / n, 2) if n else 0.0,
        "confidence_policy": {
            "confidence_threshold": config.confidence_threshold,
            "low_confidence_floor": config.low_confidence_floor,
            "definition": (
                "Per-frame pose confidence is the mean MediaPipe visibility over the 16 tracked "
                "body landmarks. A frame counts as measured when that mean reaches "
                "confidence_threshold, low-confidence between the floor and the threshold, and "
                "unusable below the floor. Individual metrics are additionally gated on the "
                "minimum visibility of the specific landmarks they depend on."
            ),
        },
    }


def _biomechanics_block(angles_3d, angles_2d, rotations) -> Dict:
    block: Dict[str, Dict] = {}
    for name, series in angles_3d.items():
        entry = summarise(series)
        entry["description"] = JOINT_ANGLE_DESCRIPTIONS.get(name, "")
        entry["image_plane_2d"] = summarise(angles_2d[name])
        block[name] = entry

    orientation: Dict[str, Dict] = {}
    for name, series in rotations.items():
        # Wrapping quantities need circular statistics; an unsigned magnitude
        # such as torso_inclination does not wrap and uses linear ones.
        period = ROTATION_PERIODS.get(name)
        entry = summarise_angular(series, period) if period else summarise(series)
        entry["description"] = ROTATION_DESCRIPTIONS.get(name, "")
        if name in ("shoulder_orientation", "hip_orientation"):
            values = np.array([m.value for m in series if m.is_measured], dtype=float)
            if values.size >= 2:
                unwrapped = unwrap_degrees(
                    np.array([m.as_float() for m in series], dtype=float)
                )
                unwrapped = unwrapped[np.isfinite(unwrapped)]
                if unwrapped.size >= 2:
                    entry["total_rotation_swept_deg"] = round(
                        float(np.max(unwrapped) - np.min(unwrapped)), 2
                    )
                    entry["rotation_swept_note"] = (
                        "Peak-to-peak of the temporally unwrapped series, so a turn through the "
                        "+/-180 boundary is counted as continuous rotation."
                    )
        orientation[name] = entry

    return {"joint_angles": block, "body_orientation": orientation}


def _movement_block(trajectories: Dict[str, Trajectory], scale: BodyScale,
                    config) -> Dict:
    movement: Dict[str, object] = {
        name: trajectory.summary(scale) for name, trajectory in trajectories.items()
    }
    dominant = None
    if config.hand == "right":
        dominant = "right_wrist"
    elif config.hand == "left":
        dominant = "left_wrist"
    movement["dominant_wrist"] = dominant
    movement["handedness_source"] = (
        "supplied via --hand" if dominant else
        "not supplied; both wrists are reported and neither is treated as dominant"
    )
    return movement


def build_report(
    metadata,
    config,
    smoothed_sequence,
    angles_3d,
    angles_2d,
    rotations,
    trajectories,
    scale: BodyScale,
    smoothing_report,
    performance: Dict,
    outputs: Dict[str, Optional[str]],
    backend_name: str,
    warnings: Sequence[str] = (),
) -> Dict:
    """Assemble the complete analysis document."""
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": metadata.to_dict(),
        "analysis_quality": _quality_block(smoothed_sequence, config),
        "biomechanics": _biomechanics_block(angles_3d, angles_2d, rotations),
        "movement": _movement_block(trajectories, scale, config),
        "body_scale": scale.to_dict(),
        "processing": {
            "pose_backend": backend_name,
            "model_complexity": config.model_complexity,
            "smoothing": smoothing_report.to_dict(),
            "performance": performance,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
        "configuration": config.to_dict(),
        "outputs": outputs,
        "warnings": list(warnings),
        "limitations": LIMITATIONS,
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _sanitise(value):
    """Replace NaN and infinity with None so the JSON stays strictly valid."""
    if isinstance(value, dict):
        return {k: _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json_report(report: Dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitise(report), handle, indent=2, default=_json_default)
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")           # headless: no display needed
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _measurement_arrays(series: List[Measurement], timestamps: np.ndarray):
    values = np.array([m.as_float() for m in series], dtype=float)
    measured = np.array([m.is_measured for m in series], dtype=bool)
    low = np.array([m.status is MeasurementStatus.LOW_CONFIDENCE for m in series], dtype=bool)
    plotted = values.copy()
    plotted[~(measured | low)] = np.nan
    return values, plotted, measured, low, timestamps


def _plot_no_data(title: str, path: Path, reason: str) -> Path:
    """
    Render an explicit "nothing was measurable" figure.

    A missing file is ambiguous: it could mean a crash, a disabled option, or
    no data.  An empty chart that states the reason is unambiguous, and it
    keeps the promised set of outputs complete without inventing a curve.
    """
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.set_axis_off()
    ax.text(0.5, 0.58, "No data to plot", ha="center", va="center",
            fontsize=15, color="#b03030", transform=ax.transAxes)
    ax.text(0.5, 0.38, reason, ha="center", va="center", fontsize=9,
            color="#555555", wrap=True, transform=ax.transAxes)
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_angle_series(name: str, series: List[Measurement], timestamps: np.ndarray,
                      path: Path, description: str = "") -> Optional[Path]:
    """Angle against time: x in seconds, y in degrees."""
    plt = _setup_matplotlib()
    values, plotted, measured, low, t = _measurement_arrays(series, timestamps)
    if not np.isfinite(plotted).any():
        logger.warning(
            "No confidently measured samples for %s; writing an empty-state plot.", name
        )
        return _plot_no_data(
            f"{name.replace('_', ' ')} angle over time", path,
            "No frame reached the confidence floor for the landmarks this angle needs.\n"
            "The joint was most likely occluded throughout the clip.",
        )

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(t, plotted, color="#1f77b4", linewidth=1.7, label="angle", zorder=3)

    # Low-confidence samples are marked, not silently blended in.
    if low.any():
        ax.scatter(t[low], values[low], s=16, color="#ff9f1c", zorder=4,
                   label="low confidence", edgecolors="none")
    # Frames with no measurement at all are shaded.
    missing = ~(measured | low)
    if missing.any():
        ymin, ymax = np.nanmin(plotted), np.nanmax(plotted)
        ax.fill_between(t, ymin, ymax, where=missing, color="#d62728", alpha=0.10,
                        step="mid", label="unavailable", zorder=1)

    if measured.any():
        finite = values[measured]
        ax.axhline(float(np.mean(finite)), color="#666666", linestyle="--", linewidth=0.9,
                   label=f"mean {np.mean(finite):.1f} deg", zorder=2)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (degrees)")
    ax.set_title(f"{name.replace('_', ' ')} angle over time")
    if description:
        fig.text(0.01, -0.02, description, fontsize=6.4, color="#555555", wrap=True, va="top")
    ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_trajectory(trajectories: Dict[str, Trajectory], names: Sequence[str],
                    title: str, path: Path, height: int, width: int) -> Optional[Path]:
    """
    Trajectory in image coordinates.

    The y axis is inverted so the plot matches what is seen on screen: image y
    grows downward, and a plot with y growing upward would show the movement
    upside down.
    """
    plt = _setup_matplotlib()
    plotted_any = False
    fig, ax = plt.subplots(figsize=(6.4, 5.0))

    palette = {
        "right_wrist": "#ff7f0e", "left_wrist": "#1f77b4",
        "hip_center": "#2ca02c", "left_foot": "#d62728", "right_foot": "#9467bd",
    }
    has_low_confidence = False
    for name in names:
        trajectory = trajectories.get(name)
        if trajectory is None:
            continue
        usable = trajectory.points_for_drawing(only_measured=False)
        measured = trajectory.points_for_drawing(only_measured=True)
        if not np.isfinite(usable).any():
            continue
        plotted_any = True
        color = palette.get(name, "#333333")

        # Draw the low-confidence path faint and dotted underneath, and the
        # confidently measured path solid on top, so the reader can see which
        # parts of the path are actually supported by good data. NaN breaks the
        # line, so a gap in the data stays a visible gap rather than a shortcut.
        if np.isfinite(usable).any() and not np.array_equal(
            np.isfinite(usable), np.isfinite(measured)
        ):
            has_low_confidence = True
            ax.plot(usable[:, 0], usable[:, 1], color=color, linewidth=1.0,
                    alpha=0.35, linestyle=":", zorder=2)

        ax.plot(measured[:, 0], measured[:, 1], color=color, linewidth=1.5, alpha=0.9,
                label=name.replace("_", " "), zorder=3)

        finite = np.flatnonzero(np.isfinite(measured).all(axis=1))
        if finite.size:
            ax.scatter(*measured[finite[0]], color=color, marker="o", s=42,
                       edgecolors="white", linewidths=1.0, zorder=5)
            ax.scatter(*measured[finite[-1]], color=color, marker="s", s=42,
                       edgecolors="white", linewidths=1.0, zorder=5)

    if not plotted_any:
        plt.close(fig)
        logger.warning("No usable data for %s; writing an empty-state plot.", title)
        return _plot_no_data(
            title, path,
            "None of the required points were confidently measured in any frame.",
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)                 # inverted: image convention
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    subtitle = "circle = start, square = end"
    if has_low_confidence:
        subtitle += "; dotted = low confidence, gaps = not measured"
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confidence_timeline(sequence, config, path: Path) -> Optional[Path]:
    """Per-frame pose confidence with the two policy thresholds marked."""
    plt = _setup_matplotlib()
    confidence = sequence.frame_confidence()
    t = sequence.timestamps

    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    ax.plot(t, confidence, color="#1f77b4", linewidth=1.5, label="pose confidence")
    ax.axhline(config.confidence_threshold, color="#2ca02c", linestyle="--", linewidth=1.0,
               label=f"MEASURED threshold ({config.confidence_threshold})")
    ax.axhline(config.low_confidence_floor, color="#d62728", linestyle=":", linewidth=1.0,
               label=f"UNAVAILABLE floor ({config.low_confidence_floor})")
    undetected = ~sequence.detected
    if undetected.any():
        ax.fill_between(t, 0, 1, where=undetected, color="#d62728", alpha=0.15, step="mid",
                        label="no pose detected")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean visibility of tracked landmarks")
    ax.set_title("Pose confidence over time")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_raw_vs_smoothed(raw_sequence, smoothed_sequence, point_name: str,
                         path: Path) -> Optional[Path]:
    """Show what smoothing actually did, so its effect is auditable."""
    plt = _setup_matplotlib()
    raw = L.point_from_array(raw_sequence.image_xy, point_name)
    smooth = L.point_from_array(smoothed_sequence.image_xy, point_name)
    t = smoothed_sequence.timestamps
    if not np.isfinite(raw).any():
        return None

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.8), sharex=True)
    for axis, index, label in zip(axes, (0, 1), ("x (pixels)", "y (pixels)")):
        axis.plot(t, raw[:, index], color="#cccccc", linewidth=2.2, label="raw", zorder=2)
        axis.plot(t, smooth[:, index], color="#1f77b4", linewidth=1.4, label="smoothed", zorder=3)
        axis.set_ylabel(label)
        axis.legend(loc="best", fontsize=7.5, framealpha=0.9)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(f"{point_name.replace('_', ' ')}: raw vs smoothed "
                      f"({smoothed_sequence.source})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rotation_series(rotations: Dict[str, List[Measurement]], timestamps: np.ndarray,
                         path: Path) -> Optional[Path]:
    """Shoulder and hip orientation with their separation underneath."""
    plt = _setup_matplotlib()
    shoulder = np.array([m.as_float() for m in rotations["shoulder_orientation"]], dtype=float)
    hip = np.array([m.as_float() for m in rotations["hip_orientation"]], dtype=float)
    separation = np.array(
        [m.as_float() for m in rotations["shoulder_hip_separation"]], dtype=float
    )
    if not np.isfinite(shoulder).any():
        return None

    shoulder_u = unwrap_degrees(shoulder)
    hip_u = unwrap_degrees(hip)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True)
    axes[0].plot(timestamps, shoulder_u, color="#1f77b4", linewidth=1.6,
                 label="shoulder line (unwrapped)")
    axes[0].plot(timestamps, hip_u, color="#2ca02c", linewidth=1.6, label="hip line (unwrapped)")
    axes[0].set_ylabel("orientation (degrees)")
    axes[0].set_title("Body orientation about the vertical axis\n"
                      "0 = square to camera facing away, -90 = right side nearer the camera")
    axes[0].legend(loc="best", fontsize=7.5, framealpha=0.9)

    axes[1].plot(timestamps, separation, color="#ff7f0e", linewidth=1.6)
    axes[1].axhline(0, color="#888888", linewidth=0.8)
    axes[1].set_ylabel("separation (degrees)")
    axes[1].set_xlabel("time (s)")
    axes[1].set_title("Shoulder-hip separation (X-factor)")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_plots(output_dir: Path, angles_3d, rotations, trajectories,
                   raw_sequence, smoothed_sequence, config, metadata) -> Dict[str, str]:
    """Write every plot and return ``{name: path}`` for the ones produced."""
    produced: Dict[str, str] = {}
    timestamps = smoothed_sequence.timestamps

    for name in ("right_elbow", "left_elbow", "right_knee", "left_knee",
                 "right_hip", "left_hip"):
        path = output_dir / f"{name}_angle.png"
        made = plot_angle_series(name, angles_3d[name], timestamps, path,
                                 JOINT_ANGLE_DESCRIPTIONS.get(name, ""))
        if made:
            produced[f"{name}_angle_plot"] = str(made)

    wrists = [n for n in ("right_wrist", "left_wrist") if n in trajectories]
    made = plot_trajectory(trajectories, wrists, "Wrist trajectory",
                           output_dir / "wrist_trajectory.png",
                           metadata.height, metadata.width)
    if made:
        produced["wrist_trajectory_plot"] = str(made)

    made = plot_trajectory(trajectories, ["hip_center"], "Hip centre trajectory",
                           output_dir / "hip_trajectory.png",
                           metadata.height, metadata.width)
    if made:
        produced["hip_trajectory_plot"] = str(made)

    made = plot_trajectory(trajectories, ["left_foot", "right_foot"], "Foot trajectories",
                           output_dir / "foot_trajectories.png",
                           metadata.height, metadata.width)
    if made:
        produced["foot_trajectories_plot"] = str(made)

    made = plot_rotation_series(rotations, timestamps, output_dir / "body_orientation.png")
    if made:
        produced["body_orientation_plot"] = str(made)

    made = plot_confidence_timeline(smoothed_sequence, config,
                                    output_dir / "pose_confidence.png")
    if made:
        produced["pose_confidence_plot"] = str(made)

    reference_point = "right_wrist" if config.hand != "left" else "left_wrist"
    made = plot_raw_vs_smoothed(raw_sequence, smoothed_sequence, reference_point,
                                output_dir / "raw_vs_smoothed_wrist.png")
    if made:
        produced["raw_vs_smoothed_plot"] = str(made)

    return produced
