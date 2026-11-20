# pyJAABA

Scores a behaviour frame by frame from DeepLabCut pose, with no video model and no
GPU: JAABA-style window features over a handful of pose quantities, a gradient-boosted
classifier, and bout post-processing. It writes BORIS-format exports so its calls can
seed or be checked against manual scoring.

It is not JAABA: no GUI, no MATLAB, and the features are computed from DLC keypoints
rather than JAABA's trackers. The name is for the method (per-frame features summarised
over several window sizes, then boosting) it borrows.

Every project-specific choice lives in one YAML (`examples/project_template.yaml`):
where the videos are, which DLC bodyparts play which role, the target, the behaviour,
the model and the post-processing.

## Track record

Built and validated on scoring prey approach in mouse prey-capture videos.
Leave-one-animal-out over 84 videos from 5 mice (379,083 frames, 420 bouts) against manual BORIS scoring:

| model | features | AP | precision | recall | F1 | bout recall |
|---|---|---|---|---|---|---|
| `approach_v1` | all 23 base features | 0.846 | 0.804 | 0.740 | 0.770 | 0.812 |
| `approach_v1_noconf` | without DLC likelihoods (`tgt_L`, `mouse_L`) | 0.837 | 0.797 | 0.733 | 0.764 | 0.798 |

On the same test, real JAABA reached F1 0.62 and DeepEthogram 0.63.

## Install / run

Needs numpy, pandas, scikit-learn, joblib and pyyaml.

```bash
git clone https://github.com/danielpollak/pyJAABA
pip install -e pyJAABA          # then `pyjaaba <command>` anywhere
# or, without installing: cd pyJAABA && python -m pyjaaba <command> --config project.yaml
```

## Commands

```bash
pyjaaba info     --config p.yaml       # resolved config + videos/DLC/BORIS files found per subject
pyjaaba features --config p.yaml       # per-frame features (+ BORIS labels) -> <output_dir>/features/
pyjaaba train    --config p.yaml --cv  # fit on every exhaustively scored video; --cv adds LOAO scores
pyjaaba predict  --config p.yaml       # -> <output_dir>/predictions/<model>/
pyjaaba verify   --config p.yaml --reference DIR [--loao]   # bit-for-bit check against a reference run
```

`--subjects`, `--dlc FILE ...`, `--name`, `--drop`, `--model` override the config for one run.

`predict` writes three files per video:

| file | contents |
|---|---|
| `<stem>_<beh>_frames.csv` | `frame, time_s, p, p_smooth, <beh>` |
| `<stem>_<beh>_bouts.csv` | one row per bout: start/stop frame and seconds, mean $p$ |
| `<stem>_<beh>.tsv` | the same bouts in BORIS's tabular-export layout (`START`/`STOP` rows) |

`summary.csv` sums up the runs and flags any video the model was trained on: a
prediction on a training video is not a test.

## Config

See `examples/project_template.yaml`. The essentials:

- **Files.** Videos are `<data_root>/<subject>/<video_include>` minus `<video_exclude>`.
  A video's DLC csv starts with its stem and contains `dlc_tag`; its BORIS export is the
  first of `annotation_names` (with `{stem}` substituted) that exists.
- **Keypoints.** `snout`, `neck`, `center`, `tailbase` are required roles. `quality`
  lists bodyparts averaged into `mouse_L`. `target` is optional: candidate target
  bodyparts, of which the most confident is used each frame. With no target, every
  target-relative feature is omitted.
- **Multi-animal DLC.** Set `dlc_individual`; the `single` (unique) bodyparts, e.g. a
  prey item, are available as targets.
- `extends: other.yaml` starts from another config; nested `keypoints`, `model`,
  `postprocess` merge key by key. Unknown keys are an error.

A trained model stores the config it was trained with; `predict` warns if the current
config would compute features differently (other DLC network, bodyparts or radii).

## Method

1. **Per-frame features.** Keypoints get a centred median (`keypoint_smoothing`, 5
   frames). Then up to 23 quantities: speed and acceleration of `center`; body
   ($\text{neck} - \text{tailbase}$) and head ($\text{snout} - \text{neck}$) lengths,
   turn rates and relative angle; forward and lateral velocity; and, with a target,
   snout-to-target distance and its rate, velocity toward the target, target bearing
   relative to body and head axes ($|\Phi|$, $\cos\Phi$, $|\Phi_\text{head}|$), bearing
   rate, target speed and relative speed; DLC likelihoods; arena position. Units are
   **pixels** and radians, so a model is tied to its camera geometry.
2. **Window features.** Each feature except arena position gets a centred mean, std,
   min and max at each radius $r$ (`window_radii`, default $2, 5, 10, 20$ frames), plus
   the change $x_{t+r} - x_{t-r}$: 443 columns with a target.
3. **Classifier.** `HistGradientBoostingClassifier` (300 iterations, learning rate 0.1,
   31 leaves, class-balanced, `random_state=0`; overridable under `model:`).
4. **Post-processing.** Centred running mean of $p$ (`smooth`), threshold, fill gaps
   shorter than `fill_gap`, drop bouts shorter than `min_bout`.

**Training set.** Videos whose BORIS export is exhaustive: `scored`, `no_behavior`, or
`empty_tsv` (watched, nothing scored, so every frame is a valid negative). A label
covers frames $[\text{START}, \text{STOP})$. Videos without an export are never trained on.

## Tests

```bash
python tests/test_generic.py    # synthetic project: renamed bodyparts, multi-animal csv, no-target, train/CV/predict
```

`verify` is the real-data check: it compares a config's features and leave-one-animal-out
predictions against arrays saved from a reference run. On the validation project above it
reproduces them exactly ($\max|\Delta p| = 0$).
