#!/usr/bin/env python3
"""
ai-tennis-poc: monocular tennis biomechanics proof of concept.

Runs the whole pipeline on one clip:

    video -> preprocessing -> pose detection -> landmarks -> confidence filtering
          -> temporal smoothing -> biomechanical calculations -> trajectories
          -> annotated video -> structured results

Usage
-----
    python analyze.py input/forehand.mp4
    python analyze.py input/forehand.mp4 --hand right
    python analyze.py input/forehand.mp4 --confidence-threshold 0.6 --no-show-foot-trails

Exit codes
----------
    0  analysis completed
    1  the clip could not be processed (bad file, missing model, unwritable output)
    2  the clip was processed but no pose was ever detected; the outputs written
       are diagnostic only and contain no measurements
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Allow running from anywhere without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analysis import report as report_module
from src.biomechanics import angles as angles_module
from src.biomechanics import normalization, rotations as rotations_module, trajectories as traj
from src.config import AnalysisConfig
from src.pose import detector as detector_module
from src.pose import smoothing as smoothing_module
from src.video.processor import VideoError, VideoReader, VideoWriter, ensure_directory
from src.visualization.overlay import OverlayRenderer, active_trail_names

logger = logging.getLogger("analyze")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze.py",
        description="Analyse body movement and joint mechanics in a short tennis clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="This tool reports objective, video-based estimates. It does not classify "
               "strokes, track the ball or racket, or judge technique.",
    )
    parser.add_argument(
        "video", nargs="?", default="input/forehand.mp4",
        help="Path to the input clip.",
    )
    parser.add_argument(
        "--hand", choices=["left", "right"], default=None,
        help="Playing hand. When given, that wrist is highlighted as dominant. "
             "When omitted, both wrists are reported equally; handedness is never guessed.",
    )
    parser.add_argument("--output-dir", default="output", help="Where results are written.")

    group = parser.add_argument_group("confidence policy")
    group.add_argument(
        "--confidence-threshold", type=float, default=0.5,
        help="Minimum landmark visibility for a metric to count as MEASURED.",
    )
    group.add_argument(
        "--low-confidence-floor", type=float, default=0.3,
        help="Below this visibility nothing is computed and the metric is UNAVAILABLE.",
    )

    group = parser.add_argument_group("pose model")
    group.add_argument(
        "--model", dest="model_complexity", choices=["lite", "full", "heavy"], default="full",
        help="MediaPipe Pose Landmarker variant. 'heavy' is most accurate and slowest.",
    )
    group.add_argument(
        "--min-detection-confidence", type=float, default=0.5,
        help="MediaPipe person-detection confidence.",
    )
    group.add_argument(
        "--min-tracking-confidence", type=float, default=0.5,
        help="MediaPipe frame-to-frame tracking confidence.",
    )

    group = parser.add_argument_group("temporal smoothing")
    group.add_argument(
        "--smoothing", dest="smoothing_method", choices=["savgol", "oneeuro", "none"],
        default="savgol", help="Filter applied to landmark trajectories over time.",
    )
    group.add_argument("--savgol-window", type=int, default=9,
                       help="Savitzky-Golay window in frames (forced odd).")
    group.add_argument("--savgol-polyorder", type=int, default=2,
                       help="Savitzky-Golay polynomial order.")
    group.add_argument("--max-interpolation-gap", type=int, default=5,
                       help="Longest run of missing frames bridged before smoothing.")
    group.add_argument("--spike-rejection", action=argparse.BooleanOptionalAction, default=True,
                       help="Remove single-frame landmark position outliers before smoothing. "
                            "These occur even at high reported visibility, so confidence "
                            "filtering alone does not catch them.")
    group.add_argument("--spike-factor", type=float, default=4.0,
                       help="Spike threshold, in multiples of the median frame-to-frame step "
                            "for that landmark. Higher rejects less.")

    group = parser.add_argument_group("visualisation")
    boolean = argparse.BooleanOptionalAction
    group.add_argument("--show-skeleton", action=boolean, default=True,
                       help="Draw the body skeleton.")
    group.add_argument("--show-angles", action=boolean, default=True,
                       help="Label joint angles on the video.")
    group.add_argument("--show-wrist-trail", action=boolean, default=True,
                       help="Draw the wrist movement trail.")
    group.add_argument("--show-hip-trail", action=boolean, default=True,
                       help="Draw the hip-centre movement trail.")
    group.add_argument("--show-foot-trails", action=boolean, default=True,
                       help="Draw the foot movement trails.")
    group.add_argument("--show-trajectories", action=boolean, default=None,
                       help="Shorthand that switches every trail on or off at once.")
    group.add_argument("--show-hud", action=boolean, default=True,
                       help="Draw the frame/timestamp/confidence panel.")
    group.add_argument("--show-orientation", action=boolean, default=True,
                       help="Draw the shoulder line, hip line and trunk vector.")
    group.add_argument("--trail-length", type=int, default=30,
                       help="Frames of trail history drawn behind each point.")
    group.add_argument("--output-scale", type=float, default=None,
                       help="Render scale for the annotated video. Default upscales small "
                            "clips toward 720 px wide for legibility.")

    group = parser.add_argument_group("outputs")
    group.add_argument("--video", dest="write_video", action=boolean, default=True,
                       help="Write the annotated MP4.")
    group.add_argument("--plots", dest="write_plots", action=boolean, default=True,
                       help="Write the PNG plots.")
    group.add_argument("--player-height", dest="player_height_m", type=float, default=None,
                       help="Player height in metres. Enables approximate metric estimates. "
                            "Without it, distances stay in pixels and body units.")
    group.add_argument("--max-frames", type=int, default=None,
                       help="Process only the first N frames (debugging aid).")
    group.add_argument("--fallback-fps", type=float, default=30.0,
                       help="FPS assumed when the container reports none.")
    group.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Console log verbosity.")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    show_wrist = args.show_wrist_trail
    show_hip = args.show_hip_trail
    show_feet = args.show_foot_trails
    if args.show_trajectories is not None:      # shorthand overrides the individual flags
        show_wrist = show_hip = show_feet = args.show_trajectories

    config = AnalysisConfig(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        confidence_threshold=args.confidence_threshold,
        low_confidence_floor=args.low_confidence_floor,
        smoothing_method=args.smoothing_method,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        max_interpolation_gap=args.max_interpolation_gap,
        spike_rejection=args.spike_rejection,
        spike_factor=args.spike_factor,
        hand=args.hand,
        player_height_m=args.player_height_m,
        show_skeleton=args.show_skeleton,
        show_angles=args.show_angles,
        show_wrist_trail=show_wrist,
        show_hip_trail=show_hip,
        show_foot_trails=show_feet,
        show_hud=args.show_hud,
        show_orientation=args.show_orientation,
        trail_length=args.trail_length,
        output_scale=args.output_scale,
        write_video=args.write_video,
        write_plots=args.write_plots,
        fallback_fps=args.fallback_fps,
        max_frames=args.max_frames,
    )
    # The annotated file is named after the input, so forehand.mp4 produces
    # annotated_forehand.mp4 as the brief specifies.
    config.annotated_video_name = f"annotated_{config.video_path.stem}.mp4"
    config.validate()
    return config


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def detect_poses(reader: VideoReader, config: AnalysisConfig):
    """Pass 1: run the pose model over every frame."""
    backend = detector_module.build_backend(config)
    selector = detector_module.MainPlayerSelector()
    builder = detector_module.SequenceBuilder(
        reader.metadata.width, reader.metadata.height, reader.metadata.fps
    )

    logger.info(
        "Pass 1/2: detecting pose with %s (%s model)", backend.name, config.model_complexity
    )
    started = time.perf_counter()
    frames_seen = 0
    try:
        for index, timestamp, frame in reader.frames(config.max_frames):
            detections = backend.detect(frame, int(round(timestamp * 1000)))
            builder.add(index, timestamp, selector.select(detections))
            frames_seen += 1
            if frames_seen % 50 == 0:
                logger.info("  ...%d frames", frames_seen)
    finally:
        backend.close()

    elapsed = time.perf_counter() - started
    sequence = builder.build()
    logger.info(
        "Pass 1/2 complete: %d frames, pose found in %d (%.1f%%), %.2fs (%.1f fps)",
        len(sequence), sequence.num_detected,
        100.0 * sequence.num_detected / max(1, len(sequence)), elapsed,
        len(sequence) / elapsed if elapsed > 0 else 0.0,
    )
    return sequence, elapsed, backend.name


def render_video(config: AnalysisConfig, metadata, sequence, angles_3d, rotations,
                 trajectories) -> tuple[Optional[Path], float, bool]:
    """Pass 2: re-decode the clip and draw the overlay onto every frame."""
    scale = config.resolve_output_scale(metadata.width)
    renderer = OverlayRenderer(config, metadata.width, metadata.height, scale)
    trails = active_trail_names(config)
    output_path = config.output_dir / config.annotated_video_name

    logger.info(
        "Pass 2/2: rendering annotated video at %dx%d (scale %.2f)",
        renderer.width, renderer.height, scale,
    )
    started = time.perf_counter()
    reencoded = False
    with VideoReader(config.video_path, config.fallback_fps) as reader, \
            VideoWriter(output_path, metadata.fps, renderer.width, renderer.height) as writer:
        for index, _timestamp, frame in reader.frames(config.max_frames):
            if index >= len(sequence):
                break
            frame_angles = {name: series[index] for name, series in angles_3d.items()}
            frame_rotations = {name: series[index] for name, series in rotations.items()}
            annotated = renderer.render(
                frame, sequence, index, frame_angles, frame_rotations, trajectories, trails
            )
            writer.write(annotated)
        writer.close(reencode_h264=True)
        reencoded = writer.reencoded
        frames_written = writer.frames_written

    elapsed = time.perf_counter() - started
    logger.info("Pass 2/2 complete: %d frames written to %s in %.2fs",
                frames_written, output_path, elapsed)
    return output_path, elapsed, reencoded


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(config: AnalysisConfig) -> int:
    wall_start = time.perf_counter()
    warnings: List[str] = []

    ensure_directory(config.output_dir)

    with VideoReader(config.video_path, config.fallback_fps) as reader:
        metadata = reader.metadata
        logger.info(
            "Input: %s | %s | %.2f fps (%s) | %d frames | %.2fs",
            metadata.path.name, metadata.resolution, metadata.fps, metadata.fps_source,
            metadata.frame_count, metadata.duration_seconds,
        )
        if metadata.fps_source == "fallback":
            warnings.append(
                f"FPS metadata was missing or implausible; {metadata.fps:.2f} fps was assumed. "
                f"Every timestamp and per-second quantity depends on that assumption."
            )
        raw_sequence, detect_seconds, backend_name = detect_poses(reader, config)

    n_frames = len(raw_sequence)
    detected = raw_sequence.num_detected

    if detected == 0:
        logger.error(
            "No person was detected in any of the %d frames. No measurements can be produced.",
            n_frames,
        )
        warnings.append(
            "No pose was detected in any frame. Every metric is UNAVAILABLE and the outputs "
            "are diagnostic only."
        )
    elif detected < n_frames:
        missing = n_frames - detected
        warnings.append(
            f"No pose was detected in {missing} of {n_frames} frames "
            f"({100.0 * missing / n_frames:.1f}%). Those frames carry no measurements."
        )
        logger.warning("Pose missing in %d of %d frames.", missing, n_frames)

    # --- smoothing -----------------------------------------------------
    smoothed_sequence, smoothing_report = smoothing_module.smooth_sequence(raw_sequence, config)

    # --- biomechanics --------------------------------------------------
    logger.info("Computing joint angles, body orientation and trajectories")
    analysis_start = time.perf_counter()
    angles_3d = angles_module.compute_joint_angles(smoothed_sequence, config)
    angles_2d = angles_module.compute_joint_angles_2d(smoothed_sequence, config)
    rotations = rotations_module.compute_rotations(smoothed_sequence, config)
    trajectories = traj.build_trajectories(smoothed_sequence, config)
    scale = normalization.compute_body_scale(smoothed_sequence, config)
    analysis_seconds = time.perf_counter() - analysis_start

    if not scale.has_body_units:
        warnings.append(
            "No frame had all four torso landmarks confidently visible, so no body-unit scale "
            "could be established. Distances are reported in pixels only."
        )

    # --- outputs -------------------------------------------------------
    outputs: Dict[str, Optional[str]] = {}
    render_seconds = 0.0
    reencoded = False

    if config.write_video:
        try:
            video_path, render_seconds, reencoded = render_video(
                config, metadata, smoothed_sequence, angles_3d, rotations, trajectories
            )
            outputs["annotated_video"] = str(video_path)
        except VideoError as exc:
            logger.error("Annotated video could not be written: %s", exc)
            warnings.append(f"Annotated video was not written: {exc}")
            outputs["annotated_video"] = None
    else:
        outputs["annotated_video"] = None

    frame_table = report_module.build_frame_dataframe(
        raw_sequence, smoothed_sequence, angles_3d, angles_2d, rotations, trajectories
    )
    csv_path = report_module.write_frame_csv(frame_table, config.frame_csv_path)
    outputs["frame_metrics_csv"] = str(csv_path)

    if config.write_plots:
        try:
            outputs.update(
                report_module.generate_plots(
                    config.output_dir, angles_3d, rotations, trajectories,
                    raw_sequence, smoothed_sequence, config, metadata,
                )
            )
        except Exception as exc:                      # plots must never sink a run
            logger.warning("Plot generation failed: %s", exc)
            warnings.append(f"Plots were not generated: {exc}")

    total_seconds = time.perf_counter() - wall_start
    performance = {
        "total_seconds": round(total_seconds, 3),
        "pose_detection_seconds": round(detect_seconds, 3),
        "biomechanics_seconds": round(analysis_seconds, 3),
        "video_rendering_seconds": round(render_seconds, 3),
        "frames_processed": n_frames,
        "seconds_per_frame": round(total_seconds / n_frames, 4) if n_frames else None,
        "detection_seconds_per_frame": round(detect_seconds / n_frames, 4) if n_frames else None,
        "effective_fps": round(n_frames / total_seconds, 2) if total_seconds > 0 else None,
        "detection_fps": round(n_frames / detect_seconds, 2) if detect_seconds > 0 else None,
        "realtime_factor": round((n_frames / metadata.fps) / total_seconds, 3)
        if total_seconds > 0 and metadata.fps > 0 else None,
        "annotated_video_reencoded_h264": reencoded,
    }

    document = report_module.build_report(
        metadata=metadata, config=config, smoothed_sequence=smoothed_sequence,
        angles_3d=angles_3d, angles_2d=angles_2d, rotations=rotations,
        trajectories=trajectories, scale=scale, smoothing_report=smoothing_report,
        performance=performance, outputs=outputs, backend_name=backend_name,
        warnings=warnings,
    )
    json_path = report_module.write_json_report(document, config.analysis_json_path)
    outputs["analysis_json"] = str(json_path)

    print_summary(document, config, outputs)
    return 2 if detected == 0 else 0


def print_summary(document: Dict, config: AnalysisConfig, outputs: Dict) -> None:
    """Short human-readable recap on stdout."""
    quality = document["analysis_quality"]
    performance = document["processing"]["performance"]
    video = document["video"]

    print()
    print("=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"  Clip           : {video['filename']}  {video['resolution']}  "
          f"{video['fps']} fps  {video['duration_seconds']}s  {video['frame_count']} frames")
    print(f"  Pose detected  : {quality['frames_with_pose_detected']}/"
          f"{quality['frames_analysed']} frames")
    average = quality["average_pose_confidence"]
    print(f"  Mean confidence: {average if average is not None else 'n/a'}")
    print(f"  Usable frames  : {quality['usable_frame_percentage']}% at threshold "
          f"{config.confidence_threshold}")
    print(f"  Processing     : {performance['total_seconds']}s total, "
          f"{performance['seconds_per_frame']}s/frame, "
          f"{performance['effective_fps']} fps effective")

    print("\n  Joint angles (MEASURED frames only, degrees):")
    for name, stats in document["biomechanics"]["joint_angles"].items():
        if stats.get("mean") is None:
            print(f"    {name:<14} unavailable ({stats['frames_measured']} measured frames)")
        else:
            print(f"    {name:<14} min {stats['min']:6.1f}   mean {stats['mean']:6.1f}   "
                  f"max {stats['max']:6.1f}   ({stats['coverage_percentage']}% coverage)")

    print("\n  Body orientation (degrees):")
    for name in ("shoulder_orientation", "hip_orientation", "shoulder_hip_separation",
                 "torso_inclination"):
        stats = document["biomechanics"]["body_orientation"].get(name, {})
        if stats.get("mean") is None:
            print(f"    {name:<26} unavailable")
        else:
            print(f"    {name:<26} min {stats['min']:7.1f}   mean {stats['mean']:7.1f}   "
                  f"max {stats['max']:7.1f}")

    print("\n  Movement (image-plane pixels):")
    for name in ("right_wrist", "left_wrist", "hip_center", "left_foot", "right_foot"):
        stats = document["movement"].get(name, {})
        if stats.get("path_length_px") is None:
            print(f"    {name:<12} unavailable")
        else:
            speed = stats.get("speed_px_per_s", {})
            print(f"    {name:<12} path {stats['path_length_px']:8.1f} px   "
                  f"peak speed {speed.get('max', float('nan')):8.1f} px/s   "
                  f"({stats['coverage_percentage']}% coverage)")

    if document["warnings"]:
        print("\n  Warnings:")
        for warning in document["warnings"]:
            print(f"    - {warning}")

    print("\n  Outputs:")
    for key, value in outputs.items():
        if value:
            print(f"    {key:<26} {value}")

    print("\n  Measurements are video-based estimates, not motion capture. "
          "\n  No stroke classification, ball/racket tracking or technique scoring "
          "is performed.")
    print("=" * 72)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # MediaPipe and absl are noisy on stderr; keep the console readable.
    logging.getLogger("absl").setLevel(logging.ERROR)

    try:
        config = config_from_args(args)
    except ValueError as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    try:
        return run(config)
    except VideoError as exc:
        logger.error("%s", exc)
        return 1
    except detector_module.ModelUnavailableError as exc:
        logger.error("%s", exc)
        return 1
    except detector_module.PoseBackendError as exc:
        logger.error("Pose backend failure: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
