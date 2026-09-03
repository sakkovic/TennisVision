"""
Frame annotation.

Design rules followed here, because an unreadable overlay is a useless one:

* **Confidence is visible, not hidden.**  Confidently measured joints are drawn
  solid; low-confidence joints are drawn hollow and dimmed; anything
  unavailable is simply not drawn.  A viewer can therefore see at a glance
  where the tracking is trustworthy, and nothing on screen implies a
  measurement that was not made.
* **Trails fade.**  Only the most recent ``trail_length`` frames are shown, and
  older points are blended out, so the frame does not silt up with history.
* **Left and right are colour coded** consistently: cool colours for the left
  side of the body, warm for the right.
* **Text is always outlined**, so it stays legible over both a bright court and
  a dark crowd.

Every element can be switched off from the configuration, and each drawing
routine is independent, so a future UI can compose its own subset.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..biomechanics.angles import JOINT_ANGLE_DEFINITIONS, OVERLAY_ANGLES
from ..measurement import Measurement, MeasurementStatus
from ..pose import landmarks as L

# --- Colours (BGR) ---------------------------------------------------------
COLOR_BY_GROUP: Dict[str, Tuple[int, int, int]] = {
    "torso": (210, 210, 210),
    "left_arm": (255, 190, 60),
    "right_arm": (60, 170, 255),
    "left_leg": (255, 140, 90),
    "right_leg": (70, 120, 255),
    "head": (190, 190, 190),
}

TRAIL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "right_wrist": (60, 220, 255),
    "left_wrist": (255, 210, 60),
    "hip_center": (110, 255, 140),
    "left_foot": (235, 130, 255),
    "right_foot": (150, 255, 255),
}

TRAIL_LABELS: Dict[str, str] = {
    "right_wrist": "R wrist",
    "left_wrist": "L wrist",
    "hip_center": "Hip centre",
    "left_foot": "L foot",
    "right_foot": "R foot",
}

STATUS_COLORS: Dict[MeasurementStatus, Tuple[int, int, int]] = {
    MeasurementStatus.MEASURED: (120, 230, 120),
    MeasurementStatus.LOW_CONFIDENCE: (80, 200, 255),
    MeasurementStatus.UNAVAILABLE: (110, 110, 240),
}

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DIM = (150, 150, 150)

FADE_BANDS = 6          # quantisation of the trail fade, keeps blending cheap
MIN_TRAIL_ALPHA = 0.15


def draw_text(image: np.ndarray, text: str, origin: Tuple[int, int], scale: float,
              color: Tuple[int, int, int] = WHITE, thickness: int = 1,
              outline: bool = True) -> None:
    """Text with a dark outline so it survives any background."""
    if outline:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK,
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


class OverlayRenderer:
    """
    Draws one annotated frame at a time.

    ``scale`` lets the annotated output be rendered larger than the source, so
    a small input clip still produces a readable video.  Landmark coordinates
    are multiplied by the same factor, so nothing shifts.
    """

    def __init__(self, config, width: int, height: int, scale: float = 1.0):
        self.config = config
        self.scale = float(scale)
        self.width = int(round(width * self.scale))
        self.height = int(round(height * self.scale))

        # Size everything relative to the output, so a 4K clip does not get
        # hairline strokes and a 480p clip does not get a giant HUD.
        reference = max(self.height, 240)
        self.line_thickness = max(1, int(round(reference / 260)))
        self.joint_radius = max(2, int(round(reference / 175)))
        self.font_scale = max(0.34, reference / 1500.0)
        self.small_font_scale = self.font_scale * 0.85
        self.trail_thickness = max(1, int(round(config.trail_thickness * self.scale * 0.6)))

        self._trails: Dict[str, Deque[Optional[Tuple[int, int]]]] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _to_pixel(self, point: np.ndarray) -> Optional[Tuple[int, int]]:
        if point is None or not np.all(np.isfinite(point)):
            return None
        return (int(round(float(point[0]) * self.scale)),
                int(round(float(point[1]) * self.scale)))

    def _status_for(self, visibility: float) -> MeasurementStatus:
        if visibility >= self.config.confidence_threshold:
            return MeasurementStatus.MEASURED
        if visibility >= self.config.low_confidence_floor:
            return MeasurementStatus.LOW_CONFIDENCE
        return MeasurementStatus.UNAVAILABLE

    def _weakest_status(self, *visibilities: float) -> MeasurementStatus:
        """The least trustworthy status among several landmarks."""
        order = [MeasurementStatus.MEASURED,
                 MeasurementStatus.LOW_CONFIDENCE,
                 MeasurementStatus.UNAVAILABLE]
        return max((self._status_for(v) for v in visibilities), key=order.index)

    @staticmethod
    def _dim(color: Tuple[int, int, int], factor: float = 0.55) -> Tuple[int, int, int]:
        return tuple(int(c * factor) for c in color)

    # ------------------------------------------------------------------
    # skeleton
    # ------------------------------------------------------------------
    def draw_skeleton(self, image: np.ndarray, image_xy: np.ndarray,
                      visibility: np.ndarray) -> None:
        """Bones first, then joints on top, so joints are never overdrawn."""
        for group, edges in L.SKELETON_GROUPS.items():
            color = COLOR_BY_GROUP[group]
            for a_name, b_name in edges:
                ia, ib = L.IDX[a_name], L.IDX[b_name]
                pa = self._to_pixel(image_xy[ia])
                pb = self._to_pixel(image_xy[ib])
                if pa is None or pb is None:
                    continue
                # A bone is only as trustworthy as its weaker endpoint.
                weakest = self._weakest_status(float(visibility[ia]), float(visibility[ib]))
                if weakest is MeasurementStatus.UNAVAILABLE:
                    continue
                if weakest is MeasurementStatus.LOW_CONFIDENCE:
                    cv2.line(image, pa, pb, self._dim(color), max(1, self.line_thickness - 1),
                             cv2.LINE_AA)
                else:
                    cv2.line(image, pa, pb, color, self.line_thickness, cv2.LINE_AA)

        for name in L.JOINT_LANDMARKS:
            idx = L.IDX[name]
            point = self._to_pixel(image_xy[idx])
            if point is None:
                continue
            status = self._status_for(float(visibility[idx]))
            if status is MeasurementStatus.UNAVAILABLE:
                continue
            if status is MeasurementStatus.MEASURED:
                cv2.circle(image, point, self.joint_radius, WHITE, -1, cv2.LINE_AA)
                cv2.circle(image, point, self.joint_radius, BLACK, 1, cv2.LINE_AA)
            else:
                # Hollow ring: visibly different from a confident joint.
                cv2.circle(image, point, self.joint_radius, DIM, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # orientation
    # ------------------------------------------------------------------
    def draw_orientation(self, image: np.ndarray, image_xy: np.ndarray,
                         visibility: np.ndarray) -> None:
        """Emphasise the shoulder and hip lines whose rotation we report."""
        for left, right, color in (
            ("left_shoulder", "right_shoulder", (255, 255, 120)),
            ("left_hip", "right_hip", (150, 255, 200)),
        ):
            ia, ib = L.IDX[left], L.IDX[right]
            pa, pb = self._to_pixel(image_xy[ia]), self._to_pixel(image_xy[ib])
            if pa is None or pb is None:
                continue
            if min(float(visibility[ia]), float(visibility[ib])) < self.config.low_confidence_floor:
                continue
            cv2.line(image, pa, pb, color, max(1, self.line_thickness), cv2.LINE_AA)
            for point in (pa, pb):
                cv2.circle(image, point, max(2, self.joint_radius - 1), color, -1, cv2.LINE_AA)

        # Trunk vector, hip midpoint -> shoulder midpoint.
        hip = self._to_pixel(L.point_from_array(image_xy, "hip_center"))
        shoulder = self._to_pixel(L.point_from_array(image_xy, "shoulder_center"))
        if hip is not None and shoulder is not None:
            cv2.arrowedLine(image, hip, shoulder, (220, 220, 120),
                            max(1, self.line_thickness - 1), cv2.LINE_AA, tipLength=0.12)

    # ------------------------------------------------------------------
    # angles
    # ------------------------------------------------------------------
    def draw_angles(self, image: np.ndarray, image_xy: np.ndarray,
                    angles: Dict[str, Measurement], names: Sequence[str] = OVERLAY_ANGLES) -> None:
        """
        Label the vertex of each selected joint with its angle.

        Only MEASURED and LOW_CONFIDENCE angles are labelled; an unavailable
        angle gets no text at all rather than a placeholder that could be
        misread as a number.
        """
        for name in names:
            measurement = angles.get(name)
            if measurement is None or not measurement.is_usable:
                continue
            _, vertex_name, _ = JOINT_ANGLE_DEFINITIONS[name]
            vertex = self._to_pixel(image_xy[L.IDX[vertex_name]])
            if vertex is None:
                continue

            low = measurement.status is MeasurementStatus.LOW_CONFIDENCE
            color = (80, 200, 255) if low else (255, 255, 255)
            label = f"{measurement.value:.0f}"
            if low:
                label += "?"

            # Nudge the label away from the joint, and keep it inside the frame.
            offset_x = int(12 * self.scale)
            offset_y = int(-8 * self.scale)
            x = min(max(vertex[0] + offset_x, 2), self.width - int(38 * self.scale))
            y = min(max(vertex[1] + offset_y, int(12 * self.scale)), self.height - 4)

            cv2.circle(image, vertex, max(2, self.joint_radius - 1), color, 1, cv2.LINE_AA)
            draw_text(image, label, (x, y), self.font_scale, color, 1)

    # ------------------------------------------------------------------
    # trails
    # ------------------------------------------------------------------
    def update_trails(self, trajectories: Dict[str, object], frame_index: int,
                      active_names: Sequence[str]) -> None:
        """Push this frame's point for each active trail (None when missing)."""
        for name in active_names:
            trajectory = trajectories.get(name)
            if trajectory is None:
                continue
            buffer = self._trails.setdefault(
                name, deque(maxlen=max(2, self.config.trail_length))
            )
            # Only confidently measured positions join a trail. A trail is a
            # claim about where a joint travelled, and drawing a low-confidence
            # estimate as part of that path would be exactly the kind of
            # misleading annotation this project must avoid. Rejected frames
            # leave a real gap in the trail rather than a straight shortcut.
            measured = trajectory.statuses[frame_index] is MeasurementStatus.MEASURED
            point = self._to_pixel(trajectory.xy_px[frame_index]) if measured else None
            buffer.append(point)

    def draw_trails(self, image: np.ndarray, active_names: Sequence[str]) -> None:
        """
        Draw the fading movement trails.

        Segments are bucketed into a handful of opacity bands and each band is
        blended once, which gives a smooth fade without one blend per segment.
        A break in the trail (a frame where the point was unavailable) leaves a
        real gap: the trail is never drawn straight through missing data.
        """
        bands: List[List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int], int]]]
        bands = [[] for _ in range(FADE_BANDS)]

        for name in active_names:
            buffer = self._trails.get(name)
            if not buffer or len(buffer) < 2:
                continue
            color = TRAIL_COLORS.get(name, (200, 200, 200))
            points = list(buffer)
            n = len(points)
            for i in range(1, n):
                p0, p1 = points[i - 1], points[i]
                if p0 is None or p1 is None:
                    continue
                recency = i / (n - 1)                     # 0 = oldest, 1 = newest
                band = min(FADE_BANDS - 1, int(recency * FADE_BANDS))
                thickness = max(1, int(round(self.trail_thickness * (0.45 + 0.55 * recency))))
                bands[band].append((p0, p1, color, thickness))

        for band_index, segments in enumerate(bands):
            if not segments:
                continue
            alpha = MIN_TRAIL_ALPHA + (1.0 - MIN_TRAIL_ALPHA) * (
                (band_index + 1) / FADE_BANDS
            )
            layer = image.copy()
            for p0, p1, color, thickness in segments:
                cv2.line(layer, p0, p1, color, thickness, cv2.LINE_AA)
            cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, dst=image)

        # A solid head marker on the newest point of each trail.
        for name in active_names:
            buffer = self._trails.get(name)
            if not buffer:
                continue
            newest = buffer[-1]
            if newest is None:
                continue
            color = TRAIL_COLORS.get(name, (200, 200, 200))
            cv2.circle(image, newest, max(2, self.joint_radius - 1), color, -1, cv2.LINE_AA)
            cv2.circle(image, newest, max(2, self.joint_radius - 1), BLACK, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def draw_hud(self, image: np.ndarray, frame_index: int, timestamp: float,
                 detected: bool, pose_confidence: float, total_frames: int,
                 rotations: Optional[Dict[str, Measurement]] = None,
                 active_trails: Sequence[str] = ()) -> None:
        """Frame counter, timestamp, confidence readout, legend."""
        pad = int(8 * self.scale)
        line_height = int(round(17 * self.scale * max(0.85, self.font_scale / 0.45)))
        rows = 3
        if rotations:
            rows += 1
        panel_w = int(min(self.width - 2 * pad, 232 * self.scale))
        panel_h = pad + rows * line_height + int(4 * self.scale)

        panel = image[pad:pad + panel_h, pad:pad + panel_w]
        if panel.size:
            darkened = (panel.astype(np.float32) * 0.35).astype(np.uint8)
            image[pad:pad + panel_h, pad:pad + panel_w] = darkened

        x = pad + int(7 * self.scale)
        y = pad + line_height - int(4 * self.scale)

        draw_text(image, f"Frame {frame_index + 1}/{total_frames}   t={timestamp:5.2f}s",
                  (x, y), self.font_scale, WHITE, 1)
        y += line_height

        if detected:
            status = (MeasurementStatus.MEASURED
                      if pose_confidence >= self.config.confidence_threshold
                      else MeasurementStatus.LOW_CONFIDENCE
                      if pose_confidence >= self.config.low_confidence_floor
                      else MeasurementStatus.UNAVAILABLE)
            label = f"Pose conf {pose_confidence:.2f}  {status.value}"
        else:
            status = MeasurementStatus.UNAVAILABLE
            label = "NO POSE DETECTED"
        color = STATUS_COLORS[status]
        cv2.circle(image, (x + int(4 * self.scale), y - int(4 * self.scale)),
                   max(2, int(3.5 * self.scale)), color, -1, cv2.LINE_AA)
        draw_text(image, label, (x + int(14 * self.scale), y), self.small_font_scale, color, 1)
        y += line_height

        if rotations:
            separation = rotations.get("shoulder_hip_separation")
            inclination = rotations.get("torso_inclination")
            parts = []
            if separation is not None and separation.is_usable:
                parts.append(f"Sh-Hip sep {separation.value:+.0f}")
            if inclination is not None and inclination.is_usable:
                parts.append(f"Trunk {inclination.value:.0f} from vert")
            draw_text(image, "  ".join(parts) if parts else "Orientation unavailable",
                      (x, y), self.small_font_scale, (220, 220, 220), 1)
            y += line_height

        draw_text(image, "video-based estimates, not motion capture",
                  (x, y), self.small_font_scale * 0.92, (185, 185, 185), 1)

        if active_trails:
            self._draw_legend(image, active_trails)

    def _draw_legend(self, image: np.ndarray, active_trails: Sequence[str]) -> None:
        pad = int(8 * self.scale)
        line_height = int(round(14 * self.scale * max(0.85, self.font_scale / 0.45)))
        # Anchor on the baseline of the LAST row so the final entry sits inside
        # the frame; anchoring on the first row pushed it off the bottom edge.
        y = self.height - pad - line_height * (len(active_trails) - 1)
        for name in active_trails:
            color = TRAIL_COLORS.get(name, (200, 200, 200))
            cv2.line(image, (pad, y - int(4 * self.scale)),
                     (pad + int(16 * self.scale), y - int(4 * self.scale)),
                     color, max(2, self.trail_thickness), cv2.LINE_AA)
            draw_text(image, TRAIL_LABELS.get(name, name),
                      (pad + int(21 * self.scale), y), self.small_font_scale * 0.92,
                      color, 1)
            y += line_height

    # ------------------------------------------------------------------
    # frame composition
    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, sequence, frame_index: int,
               angles: Dict[str, Measurement], rotations: Dict[str, Measurement],
               trajectories: Dict[str, object], active_trails: Sequence[str]) -> np.ndarray:
        """Compose every enabled layer onto a copy of the source frame."""
        if self.scale != 1.0:
            image = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            image = frame.copy()

        image_xy = sequence.image_xy[frame_index]
        visibility = sequence.visibility[frame_index]
        detected = bool(sequence.detected[frame_index])

        self.update_trails(trajectories, frame_index, active_trails)
        if active_trails:
            self.draw_trails(image, active_trails)

        if detected:
            if self.config.show_skeleton:
                self.draw_skeleton(image, image_xy, visibility)
            if self.config.show_orientation:
                self.draw_orientation(image, image_xy, visibility)
            if self.config.show_angles:
                self.draw_angles(image, image_xy, angles)

        if self.config.show_hud:
            confidence = float(sequence.frame_confidence()[frame_index])
            self.draw_hud(
                image, frame_index, float(sequence.timestamps[frame_index]),
                detected, confidence, len(sequence), rotations, active_trails,
            )
        return image


def active_trail_names(config) -> List[str]:
    """Which trails the current configuration wants drawn."""
    names: List[str] = []
    if config.show_wrist_trail:
        if config.hand == "right":
            names.append("right_wrist")
        elif config.hand == "left":
            names.append("left_wrist")
        else:
            names.extend(["right_wrist", "left_wrist"])
    if config.show_hip_trail:
        names.append("hip_center")
    if config.show_foot_trails:
        names.extend(["left_foot", "right_foot"])
    return names
