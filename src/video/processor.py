"""
Video reading and writing.

Everything that can go wrong with a video file is handled here so the rest of
the pipeline can assume it is working with a valid stream: a missing file, a
codec OpenCV cannot decode, absent or nonsensical FPS metadata, a truncated
file whose declared frame count is a lie, and an output directory that cannot
be created.

Encoding note
-------------
OpenCV writes MP4 through the ``mp4v`` (MPEG-4 Part 2) fourcc, which is
reliable everywhere but is not playable in most browsers and some players.
When an ffmpeg binary is available the finished file is re-encoded to H.264 +
yuv420p, which plays essentially anywhere.  The re-encode is a bonus, not a
requirement: if ffmpeg is missing the ``mp4v`` file is kept and the report says
so rather than failing the run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# FPS values outside this range are treated as broken metadata.
MIN_PLAUSIBLE_FPS = 1.0
MAX_PLAUSIBLE_FPS = 1000.0


class VideoError(RuntimeError):
    """Raised when a video cannot be opened, read or written."""


# Extensions we recognise as video containers.
VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".ogv", ".ogg",
    ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".3gp",
}


def looks_like_video(path: Path) -> bool:
    """
    Sniff the first bytes of a file for a known video container signature.

    This guard exists because OpenCV is surprisingly willing to "open" files
    that are not videos at all: handed a text file it reports a valid capture
    and hands back rendered frames.  Without this check a mistyped path would
    produce a confident-looking analysis of nonsense, which is exactly the
    failure mode this project must not have.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return False
    if len(head) < 12:
        return False

    if head[4:8] == b"ftyp":                       # MP4 / MOV / 3GP family
        return True
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return True
    if head[:4] == b"\x1a\x45\xdf\xa3":            # Matroska / WebM
        return True
    if head[:4] == b"OggS":                        # Ogg / Theora
        return True
    if head[:3] == b"FLV":
        return True
    if head[:4] == b"\x00\x00\x01\xba":            # MPEG program stream
        return True
    if head[:4] == b"\x00\x00\x01\xb3":            # MPEG video elementary stream
        return True
    if head[0:1] == b"\x47":                       # MPEG transport stream
        return True
    if head[:4] == b"\x30\x26\xb2\x75":            # ASF / WMV
        return True
    return False


@dataclass
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    fps_source: str          # "container" or "fallback"
    frame_count_source: str  # "container" or "counted"

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict:
        return {
            "filename": self.path.name,
            "path": str(self.path),
            "fps": round(self.fps, 3),
            "fps_source": self.fps_source,
            "frame_count": self.frame_count,
            "frame_count_source": self.frame_count_source,
            "duration_seconds": round(self.duration_seconds, 3),
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
        }


class VideoReader:
    """Context-managed frame source with validated metadata."""

    def __init__(self, path: Path | str, fallback_fps: float = 30.0):
        self.path = Path(path)
        self.fallback_fps = float(fallback_fps)

        if not self.path.exists():
            raise VideoError(
                f"Input video not found: {self.path}\n"
                f"Place your clip there, or pass a different path on the command line."
            )
        if self.path.is_dir():
            raise VideoError(f"Input path is a directory, not a video file: {self.path}")
        if self.path.stat().st_size == 0:
            raise VideoError(f"Input video is empty (0 bytes): {self.path}")

        # Refuse files that are not videos before OpenCV gets a chance to
        # "succeed" on them and produce a plausible-looking analysis of junk.
        if not looks_like_video(self.path):
            if self.path.suffix.lower() not in VIDEO_EXTENSIONS:
                raise VideoError(
                    f"{self.path} does not look like a video file. Its contents match no known "
                    f"video container signature and '{self.path.suffix or 'no extension'}' is not "
                    f"a recognised video extension. Supported inputs include "
                    f"{', '.join(sorted(VIDEO_EXTENSIONS))}."
                )
            logger.warning(
                "%s has a video extension but an unrecognised container signature; "
                "attempting to decode it anyway.", self.path.name,
            )

        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise VideoError(
                f"OpenCV could not open {self.path}. The file may be corrupt, or its codec "
                f"may not be supported by this OpenCV build. Try re-encoding it, for example:\n"
                f"  ffmpeg -i \"{self.path}\" -c:v libx264 -pix_fmt yuv420p output.mp4"
            )

        self.metadata = self._read_metadata()
        # A file can open yet yield nothing; check that a real frame comes out.
        ok, first = self.capture.read()
        if not ok or first is None:
            self.capture.release()
            raise VideoError(
                f"{self.path} opened but contains no readable frames. The file is probably "
                f"truncated or corrupt."
            )
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -- metadata -------------------------------------------------------
    def _read_metadata(self) -> VideoMetadata:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            self.capture.release()
            raise VideoError(f"{self.path} reports an invalid frame size ({width}x{height}).")

        raw_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if np.isfinite(raw_fps) and MIN_PLAUSIBLE_FPS <= raw_fps <= MAX_PLAUSIBLE_FPS:
            fps, fps_source = raw_fps, "container"
        else:
            fps, fps_source = self.fallback_fps, "fallback"
            logger.warning(
                "FPS metadata missing or implausible (%.3f) in %s; assuming %.1f fps. "
                "Timestamps and any per-second quantity are only as good as that assumption.",
                raw_fps, self.path.name, fps,
            )

        raw_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if raw_count > 0:
            frame_count, count_source = raw_count, "container"
        else:
            frame_count, count_source = 0, "counted"
            logger.warning(
                "Frame count unavailable for %s; it will be determined while decoding.",
                self.path.name,
            )

        return VideoMetadata(
            path=self.path, width=width, height=height, fps=fps,
            frame_count=frame_count, fps_source=fps_source, frame_count_source=count_source,
        )

    # -- iteration ------------------------------------------------------
    def frames(self, max_frames: Optional[int] = None) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Yield ``(frame_index, timestamp_seconds, frame_bgr)``.

        Timestamps are derived from the frame index and FPS rather than read
        from the container, which keeps them monotonic and evenly spaced even
        when the container timestamps are missing or erratic.
        """
        index = 0
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            yield index, index / self.metadata.fps, frame
            index += 1

        if index == 0:
            raise VideoError(f"No frames could be decoded from {self.path}.")
        if self.metadata.frame_count != index:
            # Container metadata frequently over- or under-reports; trust the decode.
            logger.info(
                "Decoded %d frames (container declared %d) from %s.",
                index, self.metadata.frame_count, self.path.name,
            )
            self.metadata.frame_count = index
            self.metadata.frame_count_source = "counted"

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary: system PATH first, then the imageio-ffmpeg wheel."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ensure_directory(path: Path) -> Path:
    """Create a directory, converting any failure into a clear VideoError."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoError(
            f"Could not create output directory {path}: {exc}. "
            f"Check the path is writable and not in use."
        ) from exc
    if not path.is_dir():
        raise VideoError(f"Output path exists but is not a directory: {path}")
    return path


class VideoWriter:
    """MP4 writer with an optional H.264 re-encode pass."""

    def __init__(self, path: Path | str, fps: float, width: int, height: int,
                 fourcc: str = "mp4v"):
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width = int(width)
        self.height = int(height)
        ensure_directory(self.path.parent)

        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.height)
        )
        if not self.writer.isOpened():
            raise VideoError(
                f"Could not open a video writer for {self.path} "
                f"(fourcc={fourcc}, {self.width}x{self.height} @ {self.fps:.2f} fps). "
                f"The output directory may not be writable or the codec may be unavailable."
            )
        self.frames_written = 0
        self.reencoded = False
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)
        self.frames_written += 1

    def close(self, reencode_h264: bool = True) -> None:
        # Idempotent: the context manager also calls this, and re-encoding a
        # second time would waste a full transcode pass over the finished file.
        if self._closed:
            return
        self._closed = True
        self.writer.release()
        if self.frames_written == 0:
            logger.warning("No frames were written to %s.", self.path)
            return
        if reencode_h264:
            self.reencoded = self._reencode_h264()

    def _reencode_h264(self) -> bool:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            logger.info(
                "ffmpeg not found; keeping the mp4v-encoded file. It is valid, but may not "
                "play in a browser. Install ffmpeg (or pip install imageio-ffmpeg) for H.264."
            )
            return False

        temp = self.path.with_name(self.path.stem + "_h264.mp4")
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(self.path),
            # H.264 with yuv420p requires even dimensions.
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("ffmpeg re-encode failed (%s); keeping the mp4v file.", exc)
            temp.unlink(missing_ok=True)
            return False

        if result.returncode != 0 or not temp.exists() or temp.stat().st_size == 0:
            logger.warning(
                "ffmpeg re-encode failed (exit %s); keeping the mp4v file. %s",
                result.returncode, (result.stderr or "").strip()[:400],
            )
            temp.unlink(missing_ok=True)
            return False

        try:
            temp.replace(self.path)
        except OSError as exc:
            logger.warning("Could not replace %s with the H.264 version: %s", self.path, exc)
            temp.unlink(missing_ok=True)
            return False
        logger.info("Re-encoded %s to H.264 for wide playback compatibility.", self.path.name)
        return True

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
