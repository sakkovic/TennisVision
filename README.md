# ai-tennis-poc

A proof of concept for the computer-vision and biomechanics core of an AI tennis
analysis product.

Give it one short clip of one player. It returns an annotated video with a body
skeleton and movement trails, a per-frame CSV of landmarks and joint mechanics, a
structured JSON summary, and a set of plots.

**This phase measures. It does not coach.** There is no ball tracking, no racket
tracking, no stroke classification, no contact-point detection, no comparison to
professionals and no technique score. Those belong to later phases, once
tennis-specific criteria have been defined and validated. Everything here is an
objective, video-based estimate with an explicit confidence attached.

---

## 1. Requirements

| | |
|---|---|
| Python | **3.10 – 3.12** (developed and tested on 3.12) |
| OS | Windows, macOS or Linux |
| Hardware | CPU only. No GPU required. |
| ffmpeg | Optional. Used to re-encode the annotated video to H.264. |

Python 3.13+ is not recommended yet: MediaPipe wheels lag behind new releases.

## 2. Set up the environment

```bash
# from the project root
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash / cmd)
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional extras:

```bash
pip install imageio-ffmpeg   # bundles an ffmpeg binary, no system install needed
pip install pytest           # to run the test suite
```

### Pose model weights

MediaPipe 1.x removed the bundled `mp.solutions.pose` graph, so the Pose
Landmarker task file is required. **It downloads automatically on first run** into
`models/`. To fetch it manually, or to work offline:

```bash
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

Swap `full` for `lite` (faster) or `heavy` (most accurate) and pass the matching
`--model` flag.

## 3. Add your video

Put your clip in `input/`:

```
input/forehand.mp4
```

What works best:

- **One clearly visible player.** Other people may appear; the largest, most
  central, most temporally consistent person is selected.
- **The whole body in frame**, feet included. Landmarks outside the frame are
  estimated and get low confidence, which downgrades every metric built on them.
- **A few seconds** is plenty. 2–6 s is the sweet spot.
- **A side or three-quarter view.** Rotation toward or away from the camera is the
  least reliable axis for a single camera.
- **Higher frame rate is better** for fast movement. 60 fps beats 30 fps for a swing.

A short bundled sample is included so you can run the pipeline immediately, see
section 9.

## 4. Run it

```bash
python analyze.py input/forehand.mp4
```

With handedness, which highlights the dominant wrist:

```bash
python analyze.py input/forehand.mp4 --hand right
```

Common options:

```bash
# stricter confidence gate
python analyze.py input/forehand.mp4 --confidence-threshold 0.65

# most accurate pose model
python analyze.py input/forehand.mp4 --model heavy

# turn overlay layers off
python analyze.py input/forehand.mp4 --no-show-foot-trails --no-show-angles

# every trail off at once
python analyze.py input/forehand.mp4 --no-show-trajectories

# approximate metric estimates (see the caveat in section 7)
python analyze.py input/forehand.mp4 --player-height 1.85

# compare filters, or switch smoothing off entirely
python analyze.py input/forehand.mp4 --smoothing oneeuro
python analyze.py input/forehand.mp4 --smoothing none

# skip the slow parts while iterating
python analyze.py input/forehand.mp4 --no-video --max-frames 60
```

`python analyze.py --help` lists everything.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Analysis completed. |
| 1 | The clip could not be processed: file missing, not a video, unreadable codec, model unavailable, output directory not writable. |
| 2 | The clip was processed but **no pose was detected in any frame**. Outputs are written for diagnosis and contain no measurements. |

## 5. What you get

All in `output/`:

| File | Contents |
|---|---|
| `annotated_<name>.mp4` | Video with skeleton, joints, joint-angle labels, movement trails, shoulder and hip lines, and a confidence panel. |
| `frame_metrics.csv` | One row per frame: every landmark, every angle, every orientation metric, each with status and confidence. |
| `analysis.json` | Structured summary: clip metadata, analysis quality, per-metric statistics, movement summaries, processing performance, configuration, warnings and limitations. |
| `right_elbow_angle.png` and 5 more | Joint angle against time, one per joint. |
| `wrist_trajectory.png` | Wrist paths in image coordinates. |
| `hip_trajectory.png` | Hip-centre path. |
| `foot_trajectories.png` | Both foot paths. |
| `body_orientation.png` | Shoulder and hip rotation, plus their separation. |
| `pose_confidence.png` | Per-frame confidence against the policy thresholds. |
| `raw_vs_smoothed_wrist.png` | What temporal smoothing actually changed. |

A plot whose metric was never measurable is still written, as an explicit
"no data to plot" figure. A missing file would be ambiguous; an empty chart that
states the reason is not.

## 6. How the pipeline works

```
video
  -> decode + validate            src/video/processor.py
  -> pose detection per frame     src/pose/detector.py
  -> main-player selection        src/pose/detector.py
  -> spike rejection              src/pose/smoothing.py
  -> temporal smoothing           src/pose/smoothing.py
  -> confidence classification    src/measurement.py
  -> joint angles                 src/biomechanics/angles.py
  -> body orientation             src/biomechanics/rotations.py
  -> trajectories + speeds        src/biomechanics/trajectories.py
  -> scale normalisation          src/biomechanics/normalization.py
  -> annotated video              src/visualization/overlay.py
  -> CSV / JSON / plots           src/analysis/report.py
```

```
ai-tennis-poc/
├── input/                      your clips
├── output/                     generated results
├── models/                     pose model weights (downloaded on first run)
├── src/
│   ├── config.py               all tunables and toggles
│   ├── measurement.py          the MEASURED / LOW_CONFIDENCE / UNAVAILABLE contract
│   ├── pose/
│   │   ├── landmarks.py        landmark names, indices, skeleton, derived points
│   │   ├── detector.py         backend interface + MediaPipe implementation
│   │   └── smoothing.py        spike rejection, Savitzky-Golay, One Euro
│   ├── biomechanics/
│   │   ├── angles.py           the angle(A, B, C) primitive and joint angles
│   │   ├── rotations.py        shoulder/hip orientation, separation, trunk lean
│   │   ├── trajectories.py     paths, speeds, path length
│   │   └── normalization.py    body units and the optional metric estimate
│   ├── visualization/overlay.py  all drawing
│   ├── video/processor.py        reading, writing, validation
│   └── analysis/report.py        CSV, JSON, plots
├── tests/                      85 tests
├── analyze.py                  the CLI
└── requirements.txt
```

The clip is decoded twice: once to detect pose, once to render the overlay. That
keeps memory flat regardless of clip length, since raw frames are never all held
at once.

### Two passes, and why smoothing needs them

Smoothing uses a non-causal filter, which needs future frames as well as past
ones. That is only possible offline, which is why detection completes before any
measurement is computed. It is also why the numbers have no phase lag: peak
timing is preserved, which matters when the quantity of interest is a swing.

## 7. Every metric, and how it is calculated

### Coordinate systems

The pose model produces three, and they are used for different things.

| Name | What it is | Used for |
|---|---|---|
| **Image pixels** | Origin top-left, x right, y **down**. | Drawing, trajectories, on-screen tilt. |
| **Normalised** | x and y in [0, 1] relative to width and height; z a relative depth. | Storage and resolution-independent export. |
| **World** | Approximate metres, origin at the **hip midpoint**, x right, y down, z away from the camera. | Joint angles, body orientation. |

Two consequences worth knowing:

- **Angles are never computed from normalised coordinates.** On a non-square frame
  x and y are divided by different numbers, which shears the geometry and corrupts
  any angle. Pixels or world coordinates are used instead.
- **Trajectories are never computed from world coordinates.** The world origin
  rides along with the hips, so in that space the hip centre sits at (0, 0, 0)
  forever and global movement is invisible. Image coordinates carry the movement.

### Landmarks

The 33-point BlazePose topology. Sixteen are exported per frame:

both shoulders, elbows, wrists, hips, knees, ankles, heels and foot indices (toes).

Derived points, each the midpoint of two landmarks:

| Point | Definition |
|---|---|
| `hip_center` | (left_hip + right_hip) / 2 |
| `shoulder_center` | (left_shoulder + right_shoulder) / 2 |
| `left_foot` / `right_foot` | (heel + foot_index) / 2, steadier than either alone |

A derived point takes the **minimum** confidence of its parts, never the mean: a
midpoint built from one clear and one occluded landmark is an unreliable midpoint
and must say so.

### Joint angles

For three points A, B, C with B at the joint:

```
BA = A - B
BC = C - B
angle = atan2( |BA x BC| , BA . BC )      in degrees, range [0, 180]
```

`atan2` of the cross-product magnitude against the dot product is used instead of
`arccos(dot / (|BA| |BC|))` because it stays precise near 0° and 180°, where the
arccos form degrades and can return NaN when rounding pushes its argument outside
[-1, 1]. A tennis elbow spends much of a stroke near full extension, which is
exactly where that matters.

180° means fully extended, smaller means more flexed.

| Angle | Vertex | Points |
|---|---|---|
| `right_elbow` | right elbow | right_shoulder → right_elbow → right_wrist |
| `left_elbow` | left elbow | left_shoulder → left_elbow → left_wrist |
| `right_knee` | right knee | right_hip → right_knee → right_ankle |
| `left_knee` | left knee | left_hip → left_knee → left_ankle |
| `right_hip` | right hip | right_shoulder → right_hip → right_knee |
| `left_hip` | left hip | left_shoulder → left_hip → left_knee |

Each is computed twice: from world coordinates (the primary value) and in the
image plane (column `*_angle_2d`). The 2D value is a projection, so it disagrees
with the 3D value whenever the limb is not parallel to the image plane; an
extended arm pointing at the camera projects to a sharply bent angle. Both are
exported so the difference is visible rather than hidden.

### Body orientation

| Metric | Definition |
|---|---|
| `shoulder_orientation` | Rotation of the shoulder line about the vertical axis: `atan2(v_z, v_x)` where `v = right_shoulder - left_shoulder` in world coordinates. Range (-180, 180]. |
| `hip_orientation` | The same for `right_hip - left_hip`. |
| `shoulder_hip_separation` | `shoulder_orientation - hip_orientation`, wrapped to (-180, 180]. The upper-to-lower body separation, the "X-factor". |
| `torso_inclination` | Unsigned angle between the trunk vector (hip midpoint → shoulder midpoint) and true vertical. 0° = upright. |
| `torso_lateral_lean` | Signed trunk lean in the image plane. Positive = toward the right of the image. |
| `torso_forward_lean` | Signed trunk lean along the camera axis. Positive = away from the camera. |
| `shoulder_tilt` / `hip_tilt` | On-screen tilt from pixel coordinates, wrapped to (-90, 90]. Positive = right side lower in the frame. |

Reading the orientation sign:

```
   0°   square to the camera, facing away
±180°   square to the camera, facing it
  -90°  fully side-on, RIGHT shoulder nearer the camera
  +90°  fully side-on, LEFT shoulder nearer the camera
```

The orientation is kept **directed** rather than folded into a half turn. Folding
would place the discontinuity at ±90°, which is exactly where a player filmed from
the side of the court spends most of the clip, and the metric would flip sign
every few frames. Keeping it directed moves the branch cut to ±180°.

Because a wrapped angle still has a cut somewhere, the CSV also carries
`shoulder_orientation_unwrapped` and `hip_orientation_unwrapped`, made continuous
across the boundary. The plots use those, and `total_rotation_swept_deg` in the
JSON is their peak-to-peak.

**Statistics for wrapping angles are circular**, not arithmetic. The ordinary mean
of -179° and +179° is 0°, pointing the opposite way to both samples. The circular
mean `atan2(mean(sin θ), mean(cos θ))` is used instead, with `resultant_length` in
[0, 1] reporting how concentrated the samples are. A value near 0 means the body
swept a wide arc and no single mean describes it, which is normal for a stroke.
`torso_inclination` is an unsigned magnitude that does not wrap, so it keeps
ordinary statistics.

### Trajectories

Tracked for both wrists, the hip centre and both feet, in image pixels.

- **Path length**: the sum of frame-to-frame displacements, accumulated only
  across consecutive confidently measured pairs. Gaps are skipped, never bridged.
  When any transition was skipped the JSON sets `path_length_is_lower_bound` and
  reports how many.
- **Net displacement**: straight-line distance from the first to the last measured
  position.
- **Speed**: derived from smoothed coordinates, central differences inside each
  contiguous run and one-sided at its ends. Runs are never joined across a gap.

### Scale, and what "metres" means here

A single camera has no absolute scale: the same stroke filmed from twice the
distance produces half the pixel motion. Two normalisations are offered.

**Body units** (no extra assumptions). Distances divided by the player's own torso
length in pixels, the median over confidently measured frames. This cancels camera
distance and resolution and is the safest cross-clip comparison available from the
video alone.

**Approximate metres** (only with `--player-height`). Standard anthropometry puts
the shoulder joint near 0.818 of stature and the hip joint near 0.530 (Winter,
*Biomechanics and Motor Control of Human Movement*), so the trunk segment measured
here spans about `0.288 × height`:

```
metres_per_pixel = (0.288 × player_height_m) / torso_length_px
```

Every value derived this way is labelled an estimate. It assumes the trunk is
roughly perpendicular to the camera axis, and it ignores perspective entirely, so
a player moving toward the camera will drift. Without `--player-height` nothing
metric is claimed at all.

### Confidence: the core contract

Every measurement carries a status. The confidence for a metric is the **minimum**
visibility across all landmarks it depends on, so one occluded wrist takes down
the elbow angle that needs it.

| Status | Condition | Behaviour |
|---|---|---|
| `MEASURED` | Weakest landmark ≥ `--confidence-threshold` (0.50) | Reported, drawn solid, included in statistics. |
| `LOW_CONFIDENCE` | Between the floor and the threshold | Value kept in the CSV and flagged, drawn dimmed and marked `?` on the video, **excluded from summary statistics**, dotted in trajectory plots. |
| `UNAVAILABLE` | Weakest landmark < `--low-confidence-floor` (0.30), or no pose, or degenerate geometry | **No number at all.** Blank in the CSV, `null` in the JSON, nothing drawn. |

Nothing is ever interpolated into a reported measurement, carried over from a
previous frame, or defaulted to zero.

### Spike rejection

Pose models occasionally place a landmark tens of pixels away for exactly one
frame and then recover, **while still reporting high visibility**. The confidence
score describes whether the model believes a joint is visible, not whether it put
it in the right place, so confidence filtering cannot catch this, and a polynomial
smoother is not robust to it either.

A spike is identified geometrically: the point jumps away from its predecessor,
jumps back to its successor, and yet predecessor and successor are close together.
Genuine fast movement fails that last condition, because during real motion the
surrounding frames are far apart too. The threshold adapts per landmark, at
`--spike-factor` (default 4) times its own median frame-to-frame step.

Rejected samples are removed, bridged by interpolation for the filter, and their
confidence is **capped below the MEASURED threshold**, so any metric resting on a
repaired coordinate is reported as at most `LOW_CONFIDENCE`. The count and the
affected landmarks appear in the JSON. Disable with `--no-spike-rejection`.

On the bundled sample this removes 21 samples out of 3,960, all on the occluded
far side of the body.

### Temporal smoothing

Default **Savitzky-Golay**, window 9 frames, polynomial order 2. It fits a low
order polynomial over a sliding window; being non-causal it adds no phase lag and
preserves the height and timing of peaks. A quadratic passes through it unchanged,
which is why a jump or a swing is not flattened.

**One Euro** (`--smoothing oneeuro`) is also implemented: causal and adaptive, its
cutoff rising with speed. It lags more than Savitzky-Golay offline, and it is
included because it is the filter a future real-time version would need.

Raw and smoothed coordinates are both kept. The CSV carries `<landmark>_x` /
`_y` (smoothed) alongside `<landmark>_x_raw` / `_y_raw`, and the JSON reports the
median and maximum distance smoothing moved each joint.

Gaps: runs of missing frames up to `--max-interpolation-gap` (5) are bridged
linearly **for the filter only**. Leading and trailing gaps are never
extrapolated. Frames with no detection keep `detected = False` and zero
confidence, so their metrics stay `UNAVAILABLE` regardless.

### CSV columns

| Pattern | Meaning |
|---|---|
| `frame`, `timestamp`, `pose_detected`, `pose_confidence` | Frame identity and overall quality. |
| `<landmark>_x`, `_y` | Smoothed pixel coordinates. |
| `<landmark>_x_raw`, `_y_raw` | Unsmoothed pixel coordinates. |
| `<landmark>_z` | Smoothed relative depth, hip midpoint as origin, negative = nearer. |
| `<landmark>_confidence` | Visibility in [0, 1]. |
| `<metric>` | The value. Blank when not measured. |
| `<metric>_status` | `MEASURED` / `LOW_CONFIDENCE` / `UNAVAILABLE`. |
| `<metric>_confidence` | Confidence the status came from. |
| `<joint>_angle_2d` | Image-plane version of the angle. |
| `<point>_speed_px_s` | Speed from smoothed coordinates. |

## 8. Testing

```bash
pytest                     # everything, about 16 s
pytest -m "not slow"       # unit tests only, about 2 s
```

85 tests. They check the angle primitive against hand-computed geometry, the
rotation sign conventions against constructed bodies, that the confidence policy
never lets an unmeasured value through, that smoothing preserves a quadratic
exactly, that spike rejection catches a there-and-back excursion but leaves real
acceleration alone, and that the full pipeline produces every artefact. Two tests
cover the failure paths: a clip with no person exits 2 with all metrics
`UNAVAILABLE`, and a missing file exits 1.

## 9. The bundled sample clip

`input/sample_serve.mp4` is a 4.0 s, 120-frame excerpt letting you run the
pipeline before supplying your own footage:

```bash
python analyze.py input/sample_serve.mp4 --hand right
```

Source: *Maria Sharapova on Amelia Island*, Wikimedia Commons, **CC BY 2.0**,
trimmed and re-encoded. It is a **serve**, not a forehand: it is here to exercise
the pipeline, not because a serve is the target stroke. It is also a deliberately
demanding case, filmed from the player's left so the entire right side of the body
is occluded for much of the clip. The right elbow ends up measurable in only about
8% of frames, and the report says so rather than filling the gap.

## 10. Limitations

Read these before trusting a number.

**Method**

- Everything is estimated from a single ordinary video. This is not motion
  capture, and there is no marker-based ground truth anywhere in the system.
- 3D landmark positions come from a learned statistical model of human body shape.
  Depth along the camera axis is its least reliable output, so `torso_forward_lean`
  and rotation toward or away from the camera carry the largest error.
- No camera calibration or perspective correction. Pixel distances are comparable
  only within one clip, and a player nearer the camera registers larger pixel
  motion for identical real movement.
- Image-plane speeds understate true speed for movement along the camera axis. A
  wrist travelling straight at the camera barely moves on screen.

**Definitions that are easy to over-read**

- `hip_center` is the midpoint of two hip landmarks. It is a proxy for pelvis
  position and is **not** the body centre of mass, which requires a segmented mass
  model this project does not have.
- Joint angles are angles between reconstructed body segments. They are **not**
  clinical goniometry and must not be used for diagnosis or injury assessment.
- Metre values, when present, come from an assumed height and population-average
  ratios, measured in the image plane.

**Scope**

- No ball tracking, no racket tracking, no stroke classification, no contact-point
  detection, no professional comparison, no technique score or grade.
- One person per clip. With several people visible, a size, centrality and
  continuity heuristic picks one. This is not re-identification and it can pick
  wrong in a crowded frame.
- Handedness comes from `--hand` or is left unset. It is never inferred.
- No stroke segmentation: the clip is treated as continuous motion, so statistics
  cover the whole clip and not a phase of a swing.

## 11. Extending it

**A different pose backend.** Implement `PoseDetectorBackend` in
`src/pose/detector.py` with a single `detect(frame_bgr, timestamp_ms)` returning
`RawDetection` objects in the 33-point ordering, then register it in
`build_backend()`. Nothing downstream imports MediaPipe. A backend with a
different topology should remap into that ordering and leave unsupported points
NaN with zero visibility, and the confidence policy will report them
`UNAVAILABLE` on its own.

**A new metric.** Add the calculation, its landmark dependencies and a written
definition in the relevant `src/biomechanics/` module. Wrap it with
`build_measurement()` so it inherits the confidence contract, and it flows into
the CSV, JSON and plots automatically.

**Before this becomes the product**, the following still need building:

1. Stroke segmentation and classification, to report per-swing rather than
   per-clip.
2. Ball and racket tracking, for contact point and swing path.
3. Camera calibration from court lines, which is what turns pixels into real
   distances and speeds.
4. Validation against a reference system, to establish actual error bars.
   Everything here is uncertified until that exists.
5. Only then, tennis-specific interpretation: defined, validated criteria before
   any coaching feedback is generated.
6. Multi-person tracking with re-identification for match footage.
7. Batch processing, an API and storage, none of which is in this phase.
