"""
Measurement status vocabulary.

Every biomechanical quantity this pipeline produces is wrapped in a
:class:`Measurement`.  That wrapper is the mechanism that keeps the POC honest:
a number is only ever emitted together with the confidence that produced it and
a status saying how much that number can be trusted.

The three states required by the brief:

``MEASURED``
    Every landmark the metric depends on had a visibility score at or above
    ``confidence_threshold``.  The value is reported and used in summary
    statistics.

``LOW_CONFIDENCE``
    The weakest contributing landmark fell between ``low_confidence_floor`` and
    ``confidence_threshold``.  The value is still computed and written to the
    per-frame CSV so it can be inspected, but it is flagged, drawn differently
    in the overlay, and excluded from the aggregate statistics in the JSON
    report.

``UNAVAILABLE``
    Either the pose was not detected at all, or the weakest contributing
    landmark fell below ``low_confidence_floor``, or the geometry was
    degenerate (for example two coincident points).  No number is produced.
    ``value`` is ``None`` and stays ``None``: nothing is interpolated,
    defaulted or guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class MeasurementStatus(str, Enum):
    MEASURED = "MEASURED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class Measurement:
    """A single scalar quantity plus the evidence behind it."""

    value: Optional[float]
    status: MeasurementStatus
    confidence: float = 0.0
    unit: str = ""
    source: str = ""
    note: str = ""

    # -- convenience ----------------------------------------------------
    @property
    def is_measured(self) -> bool:
        return self.status is MeasurementStatus.MEASURED

    @property
    def is_usable(self) -> bool:
        """True when a number exists at all (MEASURED or LOW_CONFIDENCE)."""
        return self.value is not None and not np.isnan(self.value)

    def as_float(self) -> float:
        """Value for tabular export; NaN when unavailable."""
        return float("nan") if self.value is None else float(self.value)

    @classmethod
    def unavailable(cls, unit: str = "", source: str = "", note: str = "") -> "Measurement":
        return cls(
            value=None,
            status=MeasurementStatus.UNAVAILABLE,
            confidence=0.0,
            unit=unit,
            source=source,
            note=note,
        )


def classify_confidence(
    confidence: float,
    confidence_threshold: float,
    low_confidence_floor: float,
) -> MeasurementStatus:
    """Map a scalar confidence onto the three-state vocabulary."""
    if not np.isfinite(confidence) or confidence < low_confidence_floor:
        return MeasurementStatus.UNAVAILABLE
    if confidence < confidence_threshold:
        return MeasurementStatus.LOW_CONFIDENCE
    return MeasurementStatus.MEASURED


def build_measurement(
    value: Optional[float],
    confidence: float,
    confidence_threshold: float,
    low_confidence_floor: float,
    unit: str = "",
    source: str = "",
    note: str = "",
) -> Measurement:
    """
    Assemble a :class:`Measurement`, applying the confidence policy.

    A non-finite ``value`` (NaN from degenerate geometry, missing landmarks)
    always collapses to ``UNAVAILABLE`` regardless of confidence, so a
    fabricated number can never leak out of the pipeline.
    """
    status = classify_confidence(confidence, confidence_threshold, low_confidence_floor)
    if value is None or not np.isfinite(value):
        return Measurement.unavailable(unit=unit, source=source, note=note or "non-finite value")
    if status is MeasurementStatus.UNAVAILABLE:
        return Measurement.unavailable(
            unit=unit, source=source, note=note or "confidence below floor"
        )
    return Measurement(
        value=float(value),
        status=status,
        confidence=float(confidence),
        unit=unit,
        source=source,
        note=note,
    )


def summarise(measurements: Sequence[Measurement]) -> dict:
    """
    Aggregate a time series of one metric.

    Only ``MEASURED`` samples feed the statistics.  The counts of every status
    are reported alongside so a reader can immediately see how much of the clip
    the statistics actually rest on.
    """
    values = np.array(
        [m.value for m in measurements if m.is_measured and m.value is not None],
        dtype=float,
    )
    counts = {status.value: 0 for status in MeasurementStatus}
    for m in measurements:
        counts[m.status.value] += 1

    total = len(measurements)
    unit = next((m.unit for m in measurements if m.unit), "")
    source = next((m.source for m in measurements if m.source), "")

    out = {
        "unit": unit,
        "source": source,
        "frames_total": total,
        "frames_measured": counts[MeasurementStatus.MEASURED.value],
        "frames_low_confidence": counts[MeasurementStatus.LOW_CONFIDENCE.value],
        "frames_unavailable": counts[MeasurementStatus.UNAVAILABLE.value],
        "coverage_percentage": round(100.0 * counts[MeasurementStatus.MEASURED.value] / total, 2)
        if total
        else 0.0,
    }

    if values.size == 0:
        out.update(
            {
                "status": MeasurementStatus.UNAVAILABLE.value,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "range": None,
                "note": "No frame reached the MEASURED confidence threshold for this metric.",
            }
        )
        return out

    out.update(
        {
            "status": MeasurementStatus.MEASURED.value,
            "min": round(float(np.min(values)), 2),
            "max": round(float(np.max(values)), 2),
            "mean": round(float(np.mean(values)), 2),
            "median": round(float(np.median(values)), 2),
            "std": round(float(np.std(values)), 2),
            "range": round(float(np.max(values) - np.min(values)), 2),
        }
    )
    return out


def summarise_angular(measurements: Sequence[Measurement], period: float = 360.0) -> dict:
    """
    Aggregate a time series of an angle that wraps.

    An ordinary arithmetic mean is wrong for a wrapped quantity: averaging
    -179 deg and +179 deg gives 0 deg, pointing in exactly the opposite
    direction to both samples.  Orientation angles in this project sit right on
    that boundary whenever a player turns through it, so the circular mean is
    used instead:

        mean = atan2( mean(sin(theta)), mean(cos(theta)) )

    ``resultant_length`` (R, in [0, 1]) reports how concentrated the samples
    are.  R near 1 means the orientation barely changed and the mean is a good
    summary; R near 0 means the body rotated through a wide arc and no single
    mean describes it, which is normal for a stroke.

    ``min``, ``max`` and ``range`` are computed on the *unwrapped* series, so
    they describe the actual arc swept rather than the wrapping artefact.
    """
    base = summarise(measurements)
    values = np.array(
        [m.value for m in measurements if m.is_measured and m.value is not None],
        dtype=float,
    )
    if values.size == 0:
        return base

    scale = 2.0 * np.pi / period
    angles = values * scale
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    resultant = float(np.hypot(sin_mean, cos_mean))
    circular_mean = float(np.arctan2(sin_mean, cos_mean)) / scale
    # Circular standard deviation (Mardia), guarded against R = 0.
    circular_std = (
        float(np.sqrt(-2.0 * np.log(max(resultant, 1e-12)))) / scale
        if resultant > 1e-12 else None
    )

    unwrapped = np.unwrap(angles) / scale

    base.update(
        {
            "statistic_type": "circular",
            "mean": round(circular_mean, 2),
            "circular_mean": round(circular_mean, 2),
            "resultant_length": round(resultant, 4),
            "circular_std": round(circular_std, 2) if circular_std is not None else None,
            "min": round(float(np.min(unwrapped)), 2),
            "max": round(float(np.max(unwrapped)), 2),
            "range": round(float(np.max(unwrapped) - np.min(unwrapped)), 2),
            "min_max_basis": "unwrapped series (continuous across the +/-180 boundary)",
            "median": None,
            "note": (
                "Circular statistics: the arithmetic mean of a wrapped angle is not "
                "meaningful. resultant_length near 0 means the samples are spread over a "
                "wide arc and no single mean value represents them."
            ),
        }
    )
    return base
