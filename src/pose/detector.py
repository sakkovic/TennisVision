"""
Pose detection.

This module defines the boundary between "how landmarks are obtained" and
"what we do with them".  Everything downstream consumes a :class:`PoseSequence`
and knows nothing about MediaPipe, so a different backend (YOLO-Pose, RTMPose,
ViTPose, a custom PyTorch model) can be added by implementing
:class:`PoseDetectorBackend` and emitting the same arrays.

Coordinate systems produced for every frame
-------------------------------------------
``image_xy``        (T, 33, 2) pixel coordinates, origin top-left.
                    This is what gets drawn and what the trajectories use.

``normalized_xyz``  (T, 33, 3) MediaPipe image-normalised coordinates.
                    x and y are in [0, 1] relative to frame width and height;
                    z is a relative depth in roughly the same scale as x, with
                    the hip midpoint as origin.  Because x and y are normalised
                    by *different* denominators on a non-square frame, angles
                    must NOT be computed directly from these values.

``world_xyz``       (T, 33, 3) MediaPipe world landmarks: an approximate metric
                    reconstruction in metres, with the origin at the midpoint
                    of the hips, axes aligned to the image (x right, y down,
                    z away from the camera).  These are the correct input for
                    joint angles and body orientation, because they are not
                    distorted by the image aspect ratio or by perspective.
                    They are a single-camera statistical estimate, not a
                    stereo or marker-based reconstruction.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import landmarks as L

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PoseBackendError(RuntimeError):
    """Raised when a pose backend cannot be constructed or run."""


class ModelUnavailableError(PoseBackendError):
    """Raised when the model weights are missing and cannot be fetched."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class RawDetection:
    """One detected person in one frame, straight from a backend."""

    normalized_xyz: np.ndarray   # (33, 3)
    world_xyz: np.ndarray        # (33, 3)
    visibility: np.ndarray       # (33,)
    presence: np.ndarray         # (33,)

    def bbox_normalized(self) -> Tuple[float, float, float, float]:
        xy = self.normalized_xyz[:, :2]
        finite = xy[np.isfinite(xy).all(axis=1)]
        if finite.size == 0:
            return (0.0, 0.0, 0.0, 0.0)
        x0, y0 = finite.min(axis=0)
        x1, y1 = finite.max(axis=0)
        return (float(x0), float(y0), float(x1), float(y1))

    def area_normalized(self) -> float:
        x0, y0, x1, y1 = self.bbox_normalized()
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def center_normalized(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bbox_normalized()
        return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=float)

    def mean_visibility(self) -> float:
        v = self.visibility[np.isfinite(self.visibility)]
        return float(v.mean()) if v.size else 0.0


@dataclass
class PoseSequence:
    """
    Whole-clip landmark arrays.

    Frames where no person was detected are represented by NaN coordinates and
    zero visibility, never by zeros or by a copy of the previous frame.  Any
    downstream consumer that meets a NaN is expected to emit UNAVAILABLE rather
    than substitute a value.
    """

    frame_indices: np.ndarray     # (T,)  int
    timestamps: np.ndarray        # (T,)  seconds
    detected: np.ndarray          # (T,)  bool
    image_xy: np.ndarray          # (T, 33, 2) pixels
    normalized_xyz: np.ndarray    # (T, 33, 3)
    world_xyz: np.ndarray         # (T, 33, 3) metres, hip-centred
    visibility: np.ndarray        # (T, 33)
    presence: np.ndarray          # (T, 33)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    source: str = "raw"           # "raw" or "smoothed(<method>)"

    def __len__(self) -> int:
        return int(self.frame_indices.shape[0])

    @property
    def num_detected(self) -> int:
        return int(np.count_nonzero(self.detected))

    def frame_confidence(self) -> np.ndarray:
        """
        Per-frame pose confidence.

        Defined as the mean visibility over the landmarks this project actually
        tracks (see ``TRACKED_LANDMARKS``), not over all 33 points: face
        landmarks are usually highly visible and would otherwise inflate the
        score for a clip where the legs are cut off.
        """
        idx = [L.IDX[n] for n in L.TRACKED_LANDMARKS]
        conf = np.where(self.detected[:, None], self.visibility[:, idx], np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(conf, axis=1)
        return np.nan_to_num(out, nan=0.0)

    def copy_with(self, **overrides) -> "PoseSequence":
        data = {
            "frame_indices": self.frame_indices,
            "timestamps": self.timestamps,
            "detected": self.detected,
            "image_xy": self.image_xy,
            "normalized_xyz": self.normalized_xyz,
            "world_xyz": self.world_xyz,
            "visibility": self.visibility,
            "presence": self.presence,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "source": self.source,
        }
        data.update(overrides)
        return PoseSequence(**data)

    @classmethod
    def empty(cls, num_frames: int, width: int, height: int, fps: float) -> "PoseSequence":
        n = L.NUM_LANDMARKS
        return cls(
            frame_indices=np.arange(num_frames, dtype=int),
            timestamps=np.zeros(num_frames, dtype=float),
            detected=np.zeros(num_frames, dtype=bool),
            image_xy=np.full((num_frames, n, 2), np.nan, dtype=float),
            normalized_xyz=np.full((num_frames, n, 3), np.nan, dtype=float),
            world_xyz=np.full((num_frames, n, 3), np.nan, dtype=float),
            visibility=np.zeros((num_frames, n), dtype=float),
            presence=np.zeros((num_frames, n), dtype=float),
            width=width,
            height=height,
            fps=fps,
        )


class SequenceBuilder:
    """Accumulates per-frame detections into a :class:`PoseSequence`."""

    def __init__(self, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        self.fps = fps
        self._frames: List[int] = []
        self._times: List[float] = []
        self._detections: List[Optional[RawDetection]] = []

    def add(self, frame_index: int, timestamp: float, detection: Optional[RawDetection]) -> None:
        self._frames.append(frame_index)
        self._times.append(timestamp)
        self._detections.append(detection)

    def build(self) -> PoseSequence:
        n_frames = len(self._frames)
        seq = PoseSequence.empty(n_frames, self.width, self.height, self.fps)
        seq.frame_indices = np.asarray(self._frames, dtype=int)
        seq.timestamps = np.asarray(self._times, dtype=float)

        for i, det in enumerate(self._detections):
            if det is None:
                continue
            seq.detected[i] = True
            seq.normalized_xyz[i] = det.normalized_xyz
            seq.world_xyz[i] = det.world_xyz
            seq.visibility[i] = det.visibility
            seq.presence[i] = det.presence
            # Normalised -> pixels.  Coordinates outside [0, 1] are kept: they
            # mean the landmark was estimated beyond the frame edge, which is
            # information we want (partially out-of-frame player), not an error.
            seq.image_xy[i, :, 0] = det.normalized_xyz[:, 0] * self.width
            seq.image_xy[i, :, 1] = det.normalized_xyz[:, 1] * self.height
        return seq


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class PoseDetectorBackend(ABC):
    """
    Interface every pose backend must satisfy.

    Implementations are expected to return landmarks in the 33-point ordering
    of :mod:`src.pose.landmarks`.  A backend with a different topology should
    remap into that ordering and leave unsupported points as NaN with zero
    visibility, so the confidence policy naturally reports them UNAVAILABLE.
    """

    name: str = "abstract"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[RawDetection]:
        """Return every person found in this frame (possibly empty)."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def __enter__(self) -> "PoseDetectorBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# MediaPipe backend
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

DEFAULT_MODEL_DIR = Path("models")


def resolve_model_path(complexity: str = "full", model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """
    Locate the ``.task`` model file, downloading it once if necessary.

    MediaPipe >= 1.0 removed the bundled ``mp.solutions.pose`` graph, so the
    Tasks API model file is a hard requirement.  It is cached in ``models/``.
    """
    if complexity not in MODEL_URLS:
        raise ValueError(
            f"Unknown model complexity {complexity!r}; expected one of {sorted(MODEL_URLS)}"
        )
    model_dir = Path(model_dir)
    path = model_dir / f"pose_landmarker_{complexity}.task"
    if path.exists() and path.stat().st_size > 0:
        return path

    try:
        model_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelUnavailableError(
            f"Cannot create model directory {model_dir}: {exc}"
        ) from exc

    url = MODEL_URLS[complexity]
    logger.info("Pose model not found locally; downloading %s -> %s", url, path)
    tmp = path.with_suffix(".task.part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"Pose model '{complexity}' is missing from {model_dir} and could not be "
            f"downloaded from {url} ({exc}). Download it manually and place it there."
        ) from exc
    return path


class MediaPipePoseBackend(PoseDetectorBackend):
    """MediaPipe Pose Landmarker (BlazePose GHUM) in VIDEO running mode."""

    name = "mediapipe_pose_landmarker"

    def __init__(
        self,
        model_complexity: str = "full",
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_dir: Path = DEFAULT_MODEL_DIR,
    ):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - environment issue
            raise PoseBackendError(
                "mediapipe is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        self.model_path = resolve_model_path(model_complexity, model_dir)
        self.model_complexity = model_complexity

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max(1, int(num_poses)),
            min_pose_detection_confidence=float(min_detection_confidence),
            min_pose_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_segmentation_masks=False,
        )
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - depends on runtime
            raise PoseBackendError(
                f"Could not initialise the MediaPipe Pose Landmarker from "
                f"{self.model_path}: {exc}"
            ) from exc
        self._last_timestamp_ms = -1

    # -- helpers --------------------------------------------------------
    @staticmethod
    def _to_arrays(landmark_list, world_list) -> RawDetection:
        n = L.NUM_LANDMARKS
        norm = np.full((n, 3), np.nan, dtype=float)
        world = np.full((n, 3), np.nan, dtype=float)
        vis = np.zeros(n, dtype=float)
        pres = np.zeros(n, dtype=float)

        for i, lm in enumerate(landmark_list[:n]):
            norm[i] = (lm.x, lm.y, lm.z)
            vis[i] = getattr(lm, "visibility", 0.0) or 0.0
            pres[i] = getattr(lm, "presence", 0.0) or 0.0

        if world_list is not None:
            for i, lm in enumerate(world_list[:n]):
                world[i] = (lm.x, lm.y, lm.z)

        return RawDetection(
            normalized_xyz=norm, world_xyz=world, visibility=vis, presence=pres
        )

    # -- interface ------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[RawDetection]:
        import cv2

        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Pose inference failed at t=%dms: %s", timestamp_ms, exc)
            return []

        if not result.pose_landmarks:
            return []

        world_sets = result.pose_world_landmarks or []
        detections = []
        for i, lm_set in enumerate(result.pose_landmarks):
            world = world_sets[i] if i < len(world_sets) else None
            detections.append(self._to_arrays(lm_set, world))
        return detections

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Main-player selection
# ---------------------------------------------------------------------------
class MainPlayerSelector:
    """
    Picks the single subject of interest when several people are detected.

    The POC assumes one main player.  When a backend returns more than one
    person (spectators, a doubles partner, the opponent), we score candidates
    on how large they are in frame, how central they are, and how close they
    are to the person selected in the previous frame.  Size dominates, because
    the player being filmed is normally the closest person to the camera.

    This is a heuristic, and it is deliberately simple: proper multi-person
    re-identification is out of scope for this phase.
    """

    def __init__(self, area_weight: float = 1.0, center_weight: float = 0.35,
                 continuity_weight: float = 0.65):
        self.area_weight = area_weight
        self.center_weight = center_weight
        self.continuity_weight = continuity_weight
        self._previous_center: Optional[np.ndarray] = None

    def select(self, detections: Sequence[RawDetection]) -> Optional[RawDetection]:
        if not detections:
            return None
        if len(detections) == 1:
            self._previous_center = detections[0].center_normalized()
            return detections[0]

        frame_center = np.array([0.5, 0.5])
        best, best_score = None, -np.inf
        for det in detections:
            area = det.area_normalized()
            center = det.center_normalized()
            centrality = 1.0 - min(1.0, float(np.linalg.norm(center - frame_center)) / 0.7071)
            score = self.area_weight * area + self.center_weight * centrality
            if self._previous_center is not None:
                distance = float(np.linalg.norm(center - self._previous_center))
                score += self.continuity_weight * max(0.0, 1.0 - distance / 0.5)
            if score > best_score:
                best, best_score = det, score

        if best is not None:
            self._previous_center = best.center_normalized()
        return best


def build_backend(config) -> PoseDetectorBackend:
    """Factory: map a config onto a concrete backend."""
    backend = getattr(config, "backend", "mediapipe").lower()
    if backend in ("mediapipe", "mediapipe_pose", "blazepose"):
        return MediaPipePoseBackend(
            model_complexity=config.model_complexity,
            num_poses=1,
            min_detection_confidence=config.min_detection_confidence,
            min_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
    raise PoseBackendError(
        f"Unknown pose backend {backend!r}. Implement PoseDetectorBackend and "
        "register it in build_backend() to add one."
    )
