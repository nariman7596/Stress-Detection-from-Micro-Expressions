# Stress Detection from Micro-Expressions

[![tests](https://github.com/nariman7596/Stress-Detection-from-Micro-Expressions/actions/workflows/tests.yml/badge.svg)](https://github.com/nariman7596/Stress-Detection-from-Micro-Expressions/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Real-time estimation of a **0–10 psychological stress index** from facial
micro-expressions, using FACS Action Units, MediaPipe FaceMesh landmarks and dense
optical flow.

> ⚠️ **Not a medical device.** This is a research and portfolio project. It has not
> been validated against any clinical outcome, it is not a diagnostic tool, and it is
> emphatically **not a lie detector**. Facial behaviour is ambiguous: the same brow
> movement accompanies concentration, glare and fear. Read the
> [Limitations](#limitations-and-validation-status) section before drawing any
> conclusion from a number this program prints.

---

## Clinical motivation

Micro-expressions are involuntary facial movements lasting roughly **40–200 ms** —
too brief to be suppressed, and usually too brief to be noticed. They leak affect that
the person is actively concealing.

In a consultation this matters. Patients routinely mask distress: pain is
under-reported, anxiety is minimised, and "I'm fine" is delivered with a smile that
never reaches the eyes. A GP has minutes per patient and is watching a dozen other
things at once. A tool that flags *when to look more closely* — not what the patient
feels — is the clinically useful framing.

The specific signals this project tracks:

- **Suppression effort** — AU23/AU24 (lips tightened and pressed), AU17 (chin raiser).
- **Fear and worry** — AU1 (inner brow raise), AU20 (lip stretch), AU7 (lid tighten).
- **Cognitive load / negative affect** — AU4 (brow lowerer), the single most reported
  facial correlate of stress.
- **Masked affect** — AU12 (lip corner pull) **without** AU6 (cheek raise). A felt,
  Duchenne smile contracts *orbicularis oculi*; a social or masking smile does not.
  AU12 without AU6, alongside active stress AUs, is the classic concealment signature.

---

## Scientific basis

Paul Ekman and Wallace Friesen's **Facial Action Coding System** (FACS) decomposes any
facial movement into anatomically defined Action Units. The stress-relevant subset
implemented here:

| AU | Name | Muscle | Clinical reading |
|----|------|--------|------------------|
| AU1 | Inner Brow Raiser | *frontalis, pars medialis* | worry, fear, anticipatory anxiety |
| AU4 | Brow Lowerer | *corrugator supercilii* | anger, concentration, cognitive load, stress |
| AU6 | Cheek Raiser | *orbicularis oculi* | genuine (Duchenne) smile — **absence** marks a masked affect |
| AU7 | Lid Tightener | *orbicularis oculi, pars palpebralis* | fear, tension, guarding |
| AU12 | Lip Corner Puller | *zygomaticus major* | smile; without AU6 a social/masking smile |
| AU17 | Chin Raiser | *mentalis* | doubt, distress, holding back |
| AU20 | Lip Stretcher | *risorius* | fear, apprehension |
| AU23 | Lip Tightener | *orbicularis oris* | anger, suppression |
| AU24 | Lip Pressor | *orbicularis oris* | suppression, holding emotion back |
| AU43/45 | Eye Closure / Blink | *levator palpebrae* (relaxation) | fatigue, gaze avoidance, withdrawal |

Intensities are reported on the FACS **A–E scale**, mapped to `0–5`.

---

## How it works

```
  frame  ─►  FaceMesh          468/478 landmarks, ~30 ms/frame
             │
             ▼
         Canonical alignment   similarity warp (rotation + scale + translation)
             │                 into a fixed 256 px crop
             ▼
         Farneback flow        dense optical flow between consecutive aligned crops,
             │                 minus its global component → non-rigid facial motion
             ▼
         Region aggregation    mean / coherence per AU region (brow, lid, cheek,
             │                 mouth corner, lip, chin)
             ├──────────────┐
             ▼              ▼
      geometric evidence  motion evidence      burst detection (40–500 ms)
      (baseline z-score)  (directional flow)   → micro-expression events
             └──────┬───────┘
                    ▼
             AU intensities (0–5)
                    │
                    ▼
             Stress index (0–10)  +  masked-smile flag, micro-expression rate,
                                     blink rate
```

Three design decisions carry most of the weight:

**Canonical alignment before flow.** Micro-expression displacements are a fraction of
a pixel at typical webcam resolution. Without cancelling head pose and camera
distance first, the flow field is entirely head motion. A *similarity* transform is
used deliberately — it removes rigid motion while preserving the non-rigid deformation
being measured. An affine fit would absorb part of the expression itself.

**A personal baseline, not a population one.** Absolute landmark distances differ far
more between people than between expressions. Everything is scored as a robust
(median/MAD) deviation from *this subject's* neutral face, captured during a short
calibration and then adapted slowly — and only on frames the estimator considers quiet,
so a sustained expression is never absorbed into "neutral".

**Two independent evidence streams.** Geometry is stable but blind to motion smaller
than landmark jitter; flow is sensitive to brief events but blind to a held expression.
Each AU combines both, plus a bonus while a burst is open in its region — that bonus is
what makes a three-frame leak visible at all.

### The stress index

A transparent weighted sum, not a learned model — every point traces back to a named
Action Unit:

| term | max points | source |
|------|-----------|--------|
| tonic AU load | 10 | weighted mean of AU intensities (AU4 1.0, AU7/AU24 0.8, AU1/AU23 0.7, AU17/AU20 0.6, AU43 0.3; **AU6 −0.4**, AU12 −0.2) |
| masked smile | +1.0 | AU12 ≥ 1.5 while AU6 < 0.8 |
| micro-expression rate | +1.5 | saturates at 20 leaks/min |
| blink rate | +1.0 | above 20/min, saturating at 45/min |

The total is clipped to `0–10` and smoothed with a 1.5 s time constant, then banded:
`calm` < 2 ≤ `mild` < 4 ≤ `moderate` < 6 ≤ `elevated` < 8 ≤ `high`.

---

## Installation

Tested on macOS 14+ (Apple Silicon) and Linux, Python 3.10–3.12.

```bash
git clone https://github.com/nariman7596/Stress-Detection-from-Micro-Expressions.git
cd Stress-Detection-from-Micro-Expressions

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` selects one of two stacks automatically, because MediaPipe is the
binding constraint:

| your Python | MediaPipe | backend | model |
|---|---|---|---|
| 3.10 – 3.12 | 0.10.21 | legacy `solutions.face_mesh` (needs NumPy < 2) | bundled in the wheel |
| 3.13+ | 1.0.1 | Tasks `FaceLandmarker` | `face_landmarker.task` downloaded into `models/` on first run |

MediaPipe 0.10.x has **no macOS arm64 wheels above CPython 3.12**; 1.0.x ships a
pure-Python wheel that installs anywhere but exposes only the Tasks API.
`src/face_mesh.py` supports both and uses whichever is installed — the active backend
is logged at startup and shown in the overlay footer, so you can always tell which one
you are on.

On macOS, grant camera access the first time you run with a USB camera
(*System Settings → Privacy & Security → Camera*). Grant it to your **terminal app**,
not to Python, and restart the terminal afterwards.

### Troubleshooting

**`error: metadata-generation-failed` on scipy, or a missing `g95`/`gfortran`.**
You are on an old checkout. SciPy is not a dependency of this project — the spectral
analysis uses `numpy.fft`. Pull the latest `requirements.txt` and reinstall.

**`ModuleNotFoundError: No module named 'cv2'`.** The dependency install aborted part
way through, so OpenCV never got installed. Fix the failing package and re-run
`pip install -r requirements.txt`; the error above this one is the real one.

**pip tries to build NumPy or MediaPipe from source.** No wheel exists for your Python
version. Check with `python --version` — anything from 3.10 to 3.14 is supported by
the pinned sets above.

**`ResolutionImpossible` mentioning `numpy<2.3.0`.** An old checkout pinned
`opencv-python==4.12.0.88`, the one release that caps NumPy below 2.3.0 — unsatisfiable
on CPython 3.14, where no NumPy arm64 wheel exists below 2.3.2. Pull and reinstall.

---

## Usage

```bash
# USB webcam with a live preview (preferred: close-up, high face resolution)
python main.py --camera 0 --show --flip

# WiFi RTSP camera, headless, logging every frame
python main.py --camera "rtsp://admin:@192.168.1.15:554/stream1" --no-show --csv session.csv

# Re-analyse a recording and write an annotated video
python main.py --camera clip.mp4 --save annotated.mp4
```

`--camera` accepts a **device index** (`0`), an **RTSP URL**, or a **video file path**.
RTSP streams are opened over TCP with a 1-frame buffer (UDP + H.265 over WiFi produces
torn frames) and are automatically reconnected with exponential backoff when they drop.

Hold a relaxed, neutral, forward-facing expression for the first few seconds — that is
the calibration, and everything afterwards is measured against it.

### Keys (with `--show`)

| key | action |
|-----|--------|
| `q` / `Esc` | quit |
| `m` | toggle the raw landmark mesh |
| `h` | toggle the AU region heat map |
| `r` | restart the neutral-baseline calibration |

### Options

| flag | default | meaning |
|------|---------|---------|
| `--camera` | `0` | device index, RTSP URL or file path |
| `--width` / `--height` / `--fps` | — | requested capture format (best effort) |
| `--flip` | off | mirror the image, natural for a webcam |
| `--show` / `--no-show` | `--show` | preview window or headless |
| `--save PATH` | — | write the annotated video |
| `--csv PATH` | — | per-frame log (score, components, all AU intensities) |
| `--max-frames N` | — | stop after N frames |
| `--calibration` | `3.0` | seconds of neutral face for the personal baseline |
| `--micro-sensitivity` | `1.0` | > 1 catches fainter leaks, < 1 is stricter |
| `--smoothing` | `1.5` | stress index time constant, seconds |
| `--align-size` | `256` | canonical face crop size |
| `--flow-downscale` | `0.5` | optical flow resolution factor (lower = faster) |
| `--mesh` / `--no-heatmap` | — | overlay toggles |
| `--print-interval` | `2.0` | seconds between console updates (`0` disables) |

### Demo

No sample recording is committed — faces are identifiable data, and a repository is
the wrong place for them. Record your own in one command:

```bash
python main.py --camera 0 --flip --save docs/demo.mp4 --max-frames 600
```

The overlay shows the stress gauge, per-AU intensity bars (with a dot marking a live
micro-expression burst in that AU's region), an AU-region heat map on the face, the
micro-expression and blink rates, the masked-smile alert, and a ticker of recent leaks.

### CSV output

One row per processed frame: `timestamp`, `stress_score` (smoothed), `stress_instant`,
`level`, `masked_smile`, `micro_rate`, `blink_rate`, `quality`, `calibrated`, a
`contrib_AU*` column per AU (its signed points in the index) and an `AU*` column per AU
(its 0–5 intensity). The schema is identical for calibrating and scored frames, so the
file loads cleanly into pandas or the notebook.

---

## Project structure

```
├── main.py                        entry point: --camera, --show, --save, --csv
├── src/
│   ├── camera.py                  camera-agnostic capture (USB index or RTSP URL)
│   ├── face_mesh.py               FaceMesh wrapper, canonical alignment, FACS geometry
│   ├── optical_flow.py            Farneback micro-motion, burst detection, spectra
│   ├── au_estimator.py            Action Unit intensity estimation (rule-based)
│   ├── stress_scorer.py           Action Units → stress index (0–10)
│   ├── visualizer.py              real-time overlay
│   └── pipeline.py                wiring of the above into one per-frame call
├── notebooks/
│   └── 01_au_exploration.ipynb    signal inspection, AU time series, spectra
└── tests/                         160 tests, no camera or MediaPipe required
```

Dependencies are deliberately minimal: NumPy, OpenCV and MediaPipe at runtime,
plus Matplotlib for the notebook. No SciPy — the FFT work uses `numpy.fft`.

`src/pipeline.py` is the only addition to the originally planned layout: keeping the
stage wiring out of `main.py` leaves the CLI thin and makes the whole pipeline
testable with an injected stub detector.

---

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The suite runs **without a camera and without MediaPipe**: the synthetic face in
`tests/conftest.py` is built analytically, and `StressPipeline` accepts an injected
detector. It covers source parsing and credential masking, the alignment maths
(including that a *sheared* target is deliberately not matched), every FACS feature's
response direction, optical-flow accuracy against a known translation, rigid-motion
cancellation, burst detection and its rejection of sustained expressions, baseline
calibration and drift, the AU rules, the scoring weights and smoothing, and overlay
rendering.

Three bugs found this way are worth naming, because they are the kind that would have
quietly corrupted every reading rather than crashing:

- a `x or y` fallback treated a legitimate timestamp of `0.0` as missing, collapsing
  the first frame interval;
- a sustained expression was aborted mid-burst, letting its tail re-trigger as a
  spurious micro-expression;
- the CSV header was locked in from an uncalibrated first row, which lacked the
  contribution columns.

---

## Limitations and validation status

**Nothing here has been validated against labelled data.** The AU rules encode
published FACS descriptions; the thresholds are set from the measured noise floor of a
static clip, not fitted. Specifically:

- **Frame rate is the binding constraint.** A 40 ms event is 1–2 frames at 30 fps. The
  corpora in this field are recorded at 200 fps. At 30 fps only the slower half of the
  micro-expression range is resolvable at all; the notebook prints your effective
  Nyquist limit.
- **Head pose.** Out-of-plane rotation is *not* corrected — it is flagged as low
  quality and scales the intensities down. Beyond roughly 30° of yaw the readings are
  not meaningful.
- **AU6 vs AU7** both narrow the eye aperture. They are separated by the cheek-raise
  term, which is the weakest rule in the set.
- **Lighting changes** move the flow field. Global compensation removes the common
  component; a hard shadow crossing one region does not have a common component.
- **One face at a time**, and a fisheye ceiling camera (the V380) gives a face far too
  small for this — use the USB camera close up.
- **The weights are literature-informed, not fitted.** They express a defensible
  ordering (AU4 matters more than AU43), not a calibrated probability.

The honest next step is a labelled corpus — **CASME II** (247 samples, coded AUs,
200 fps), **SAMM** (159, coded AUs), **SMIC** (164, 3 classes, no AU labels), and
**MAHNOB-HCI** (video plus synchronised ECG/GSR, which would let the *stress*
construct be checked against physiology rather than against another facial label).
Section 7 of the notebook sets out the experiments in order of value.

---

## References

1. Ekman, P. & Friesen, W. V. (1978). *Facial Action Coding System: A Technique for the
   Measurement of Facial Movement.* Consulting Psychologists Press.
2. Ekman, P. (2009). *Telling Lies: Clues to Deceit in the Marketplace, Politics, and
   Marriage.* W. W. Norton.
3. Yan, W.-J. et al. (2014). CASME II: An Improved Spontaneous Micro-Expression
   Database and the Baseline Evaluation. *PLoS ONE* 9(1): e86041.
4. Li, X. et al. (2013). A Spontaneous Micro-Expression Database: Inducement,
   Collection and Baseline. *IEEE FG 2013.*
5. Davison, A. K. et al. (2018). SAMM: A Spontaneous Micro-Facial Movement Dataset.
   *IEEE Transactions on Affective Computing* 9(1), 116–129.
6. Soleymani, M. et al. (2012). A Multimodal Database for Affect Recognition and
   Implicit Tagging. *IEEE Transactions on Affective Computing* 3(1), 42–55.
7. Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion.
   *SCIA 2003*, LNCS 2749, 363–370.
8. Kartynnik, Y. et al. (2019). Real-time Facial Surface Geometry from Monocular Video
   on Mobile GPUs. *CVPR Workshops.*
9. Lugaresi, C. et al. (2019). MediaPipe: A Framework for Building Perception
   Pipelines. *arXiv:1906.08172.*
10. Soukupová, T. & Čech, J. (2016). Real-Time Eye Blink Detection Using Facial
    Landmarks. *21st Computer Vision Winter Workshop.*
11. Giannakakis, G. et al. (2017). Stress and Anxiety Detection Using Facial Cues from
    Videos. *Biomedical Signal Processing and Control* 31, 89–101.

---

## License

MIT — see [LICENSE](LICENSE).
