"""Video handling, error paths and a full end-to-end run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.config import AnalysisConfig
from src.video.processor import (
    VideoError, VideoReader, VideoWriter, ensure_directory, looks_like_video,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------
def test_config_rejects_inverted_confidence_bounds():
    with pytest.raises(ValueError, match="low_confidence_floor"):
        AnalysisConfig(confidence_threshold=0.2, low_confidence_floor=0.8).validate()


def test_config_rejects_unknown_hand():
    with pytest.raises(ValueError, match="hand"):
        AnalysisConfig(hand="ambidextrous").validate()


def test_config_rejects_unknown_smoothing():
    with pytest.raises(ValueError, match="smoothing"):
        AnalysisConfig(smoothing_method="kalman").validate()


def test_config_rejects_savgol_order_at_or_above_the_window():
    with pytest.raises(ValueError, match="savgol_window"):
        AnalysisConfig(savgol_window=5, savgol_polyorder=5).validate()


def test_config_rejects_an_implausible_player_height():
    with pytest.raises(ValueError, match="player_height"):
        AnalysisConfig(player_height_m=25.0).validate()


def test_output_scale_upscales_small_clips_only():
    config = AnalysisConfig()
    assert config.resolve_output_scale(500) == pytest.approx(1.44)
    assert config.resolve_output_scale(1920) == pytest.approx(1.0)
    assert config.resolve_output_scale(100) == pytest.approx(2.0)     # capped
    assert AnalysisConfig(output_scale=1.5).resolve_output_scale(1920) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# video error handling
# ---------------------------------------------------------------------------
def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        VideoReader(tmp_path / "nope.mp4")


def test_a_directory_is_rejected(tmp_path):
    with pytest.raises(VideoError, match="directory"):
        VideoReader(tmp_path)


def test_an_empty_file_is_rejected(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoError, match="empty"):
        VideoReader(empty)


def test_a_text_file_is_rejected_even_though_opencv_would_open_it(tmp_path):
    """
    OpenCV happily "opens" a text file and returns rendered frames. Without a
    container check the pipeline would produce a confident analysis of nothing,
    which is the single worst failure mode this project can have.
    """
    text = tmp_path / "notes.txt"
    text.write_text("this is not a video\n" * 50)
    with pytest.raises(VideoError, match="does not look like a video"):
        VideoReader(text)


def test_signature_sniffing():
    assert not looks_like_video(ROOT / "requirements.txt")
    sample = ROOT / "input" / "sample_serve.mp4"
    if sample.exists():
        assert looks_like_video(sample)


def test_output_directory_creation_failure_is_reported(tmp_path):
    blocker = tmp_path / "output"
    blocker.write_text("I am a file, not a directory")
    with pytest.raises(VideoError):
        ensure_directory(blocker)


def test_fps_fallback_when_metadata_is_missing(sample_video, monkeypatch):
    """A clip with no usable FPS must fall back and say that it did."""
    import cv2

    real_get = cv2.VideoCapture.get

    def fake_get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 0.0
        return real_get(self, prop)

    monkeypatch.setattr(cv2.VideoCapture, "get", fake_get)
    with VideoReader(sample_video, fallback_fps=25.0) as reader:
        assert reader.metadata.fps == pytest.approx(25.0)
        assert reader.metadata.fps_source == "fallback"


def test_reader_reports_real_metadata(sample_video):
    with VideoReader(sample_video) as reader:
        meta = reader.metadata
        assert meta.width > 0 and meta.height > 0
        assert meta.fps > 1.0
        assert meta.frame_count > 0
        assert meta.duration_seconds > 0


def test_writer_produces_a_playable_file(tmp_path):
    import cv2

    path = tmp_path / "out.mp4"
    with VideoWriter(path, fps=25.0, width=64, height=48) as writer:
        for i in range(10):
            frame = np.full((48, 64, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
    assert path.exists() and path.stat().st_size > 0

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    capture.release()


def test_writer_close_is_idempotent(tmp_path):
    writer = VideoWriter(tmp_path / "twice.mp4", fps=25.0, width=32, height=32)
    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.close()
    writer.close()          # must not re-encode or raise a second time


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_full_pipeline_on_the_sample_clip(sample_video, tmp_path):
    """One command, all the promised artefacts, internally consistent."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "analyze.py"), str(sample_video),
         "--hand", "right", "--output-dir", str(tmp_path), "--max-frames", "40",
         "--log-level", "WARNING"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    video = tmp_path / f"annotated_{sample_video.stem}.mp4"
    csv_path = tmp_path / "frame_metrics.csv"
    json_path = tmp_path / "analysis.json"
    for artefact in (video, csv_path, json_path):
        assert artefact.exists() and artefact.stat().st_size > 0, f"missing {artefact}"

    for plot in ("right_elbow_angle.png", "right_knee_angle.png",
                 "wrist_trajectory.png", "hip_trajectory.png",
                 "foot_trajectories.png", "pose_confidence.png"):
        assert (tmp_path / plot).exists(), f"missing plot {plot}"

    import pandas as pd
    frame = pd.read_csv(csv_path)
    assert len(frame) == 40
    for column in ("frame", "timestamp", "pose_confidence",
                   "right_elbow_angle", "left_elbow_angle",
                   "right_knee_angle", "left_knee_angle",
                   "right_hip_angle", "left_hip_angle",
                   "shoulder_orientation", "hip_orientation",
                   "shoulder_hip_separation", "torso_inclination",
                   "hip_center_x", "hip_center_y",
                   "right_wrist_x", "right_wrist_y",
                   "left_wrist_x", "left_wrist_y",
                   "right_foot_x", "right_foot_y",
                   "left_foot_x", "left_foot_y"):
        assert column in frame.columns, f"missing CSV column {column}"

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["video"]["frame_count"] > 0
    assert report["analysis_quality"]["frames_analysed"] == 40
    assert report["biomechanics"]["joint_angles"]["right_elbow"]["frames_total"] == 40
    assert report["limitations"]
    assert report["processing"]["performance"]["total_seconds"] > 0

    # Angles must be physically possible wherever they are reported.
    angles = frame["right_knee_angle"].dropna()
    assert ((angles >= 0) & (angles <= 180)).all()

    # Every unavailable metric must be blank, with no leftover number.
    unavailable = frame["right_elbow_angle_status"] == "UNAVAILABLE"
    assert frame.loc[unavailable, "right_elbow_angle"].isna().all()


@pytest.mark.slow
def test_pipeline_reports_when_no_person_is_present(tmp_path):
    """
    A clip with no person must produce diagnostic output and exit code 2,
    never fabricated measurements.
    """
    import cv2

    clip = tmp_path / "empty_scene.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (192, 144))
    rng = np.random.default_rng(0)
    for _ in range(20):
        writer.write(rng.integers(0, 60, size=(144, 192, 3), dtype=np.uint8))
    writer.release()

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, str(ROOT / "analyze.py"), str(clip),
         "--output-dir", str(out_dir), "--log-level", "ERROR"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=900,
    )
    assert result.returncode == 2, f"expected exit 2, got {result.returncode}\n{result.stderr}"

    report = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
    assert report["analysis_quality"]["frames_with_pose_detected"] == 0
    assert any("No pose was detected" in w for w in report["warnings"])
    for stats in report["biomechanics"]["joint_angles"].values():
        assert stats["mean"] is None
        assert stats["status"] == "UNAVAILABLE"

    import pandas as pd
    frame = pd.read_csv(out_dir / "frame_metrics.csv")
    assert frame["right_elbow_angle"].isna().all()
    assert (frame["right_elbow_angle_status"] == "UNAVAILABLE").all()


@pytest.mark.slow
def test_missing_input_exits_with_code_one(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "analyze.py"), str(tmp_path / "absent.mp4"),
         "--output-dir", str(tmp_path), "--log-level", "ERROR"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300,
    )
    assert result.returncode == 1
    assert "not found" in (result.stderr + result.stdout)
