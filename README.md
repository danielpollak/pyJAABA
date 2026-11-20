# pyJAABA

Scores a behaviour frame by frame from DeepLabCut pose, with no video model and no
GPU: JAABA-style window features over a handful of pose quantities, a gradient-boosted
classifier, and bout post-processing. It reads DeepLabCut tracking and BORIS
annotations, and writes its calls back out in BORIS's format so they can seed, or be
checked against, manual scoring.

It is not JAABA: no GUI, no MATLAB, and the features are computed from DLC keypoints
rather than JAABA's trackers. The name is for the method it borrows (per-frame features
summarised over several window sizes, then boosting).

Every project-specific choice lives in one YAML file (`examples/project_template.yaml`):
where the videos are, which DLC bodyparts play which role, the target, the behaviour,
the model and the post-processing.

## Track record

Built and validated on scoring prey approach in mouse prey-capture videos.
Leave-one-animal-out over 84 videos from 5 mice (379,083 frames, 420 bouts) against
manual BORIS scoring:

| model | features | AP | precision | recall | F1 | bout recall |
|---|---|---|---|---|---|---|
| `approach_v1` | all 23 base features | 0.846 | 0.804 | 0.740 | 0.770 | 0.812 |
| `approach_v1_noconf` | without DLC likelihoods (`tgt_L`, `mouse_L`) | 0.837 | 0.797 | 0.733 | 0.764 | 0.798 |

On the same test, real JAABA reached F1 0.62 and DeepEthogram 0.63.

## Install

Needs numpy, pandas, scikit-learn, joblib and pyyaml.

```bash
git clone https://github.com/danielpollak/pyJAABA
pip install -e pyJAABA          # then `pyjaaba <command>` anywhere
# or, without installing: cd pyJAABA && python -m pyjaaba <command> --config project.yaml
```

## Workflow

```bash
pyjaaba info     --config p.yaml        # 1. check the config finds your files
pyjaaba features --config p.yaml        # 2. pose -> features (+ labels from BORIS)
pyjaaba train    --config p.yaml --cv   # 3. fit a model; --cv also scores it leave-one-animal-out
pyjaaba predict  --config p.yaml        # 4. score videos -> per-frame scores, bouts, BORIS file
pyjaaba verify   --config p.yaml --reference DIR [--loao]   # optional: bit-for-bit check against a reference run
```

A flag overrides the config for that run only:

| flag | commands | meaning |
|---|---|---|
| `--subjects A B` | features, train, predict | use these subject folders instead of the config's (`--subjects` with no names: none, e.g. with `--dlc`) |
| `--dlc FILE ...` | features, predict | add individual DLC csv files outside `data_root` (`--label` names their subject) |
| `--root DIR` | features, predict | read videos from here instead of `data_root` |
| `--overwrite` | features | recompute cached features |
| `--name NAME` | train | model name instead of `model_name` |
| `--drop F ...` | train | base features to leave out instead of `drop_features` |
| `--cv` | train | also run leave-one-animal-out and save its scores |
| `--model NAME\|PATH` | predict | model to use instead of `model_name` |
| `--out DIR` | predict | output folder instead of `<output_dir>/predictions/<model>/` |

## Inputs

pyJAABA only **reads** your data folder; everything it writes goes under `output_dir`.

### Folder layout

```
<data_root>/
  <subject>/                     one folder per animal, named in `subjects`
    <stem>.avi                   a video: matched by `video_include` / `video_exclude`
    <stem>DLC_...<dlc_tag>....csv   its DeepLabCut output (required)
    <stem>.avi.tsv               its BORIS export (optional; see `annotation_names`)
```

The **video** file is only used to find the session and to name it (`<stem>` is the
file name without extension). Its frames are never read: everything comes from the DLC
csv, so the video can be missing when you give `--dlc` files directly.

### DeepLabCut csv (required)

The standard `analyze_videos` csv, one row per video frame. The row count sets the
number of frames scored.

- **Single-animal**: three header rows, `scorer` / `bodyparts` / `coords`, with
  `x`, `y`, `likelihood` per bodypart.
- **Multi-animal**: four header rows, adding `individuals`. Set `dlc_individual` to
  the animal to score (it can be left out when there is only one). Bodyparts of the
  `single` individual (unique bodyparts, e.g. a prey item) are available alongside it.

The file for `<stem>` is the one in the same folder whose name starts with `<stem>` and
contains `dlc_tag`; more than one match is an error. Every bodypart named under
`keypoints` must be present.

### BORIS tabular events export (optional; needed for training and scoring)

BORIS's *Export events > Tabular events* file (tab-separated), one per video. pyJAABA
uses these columns:

| column | use |
|---|---|
| `Behavior` | behaviour name; those listed in `behavior` / `behaviors` become labels |
| `Behavior type` | `START` / `STOP` pairs make a bout; other rows are noted as quirks |
| `Image index` | frame number of the event (if blank, the next row's value, else `Time` $\times$ `FPS`) |
| `Time` | seconds; orders the events |
| `FPS` | the video's frame rate; used for time-derivative features (`fps_default` when there is no export) |

A bout covers frames $[\text{START}, \text{STOP})$: STOP is the first frame after it.
What the export means for training:

| export | status | used for training? |
|---|---|---|
| has at least one bout of `behavior` | `scored` | yes |
| has events, none of `behavior` | `no_behavior` | yes, every frame negative |
| exists but empty (watched, nothing scored) | `empty_tsv` | yes, every frame negative |
| none | `no_tsv` | no; can still be predicted |
| (no DLC csv found) | `no_dlc` | no; skipped entirely |

Only export a file for a video you have scored **completely** for `behavior`: an
unscored stretch in a `scored` or `empty_tsv` video is learned as "not the behaviour".

### Config (YAML)

Relative paths resolve against the YAML file. Unknown keys are an error, so a typo
cannot silently fall back to a default. `extends: other.yaml` starts from another
config; its top-level keys are replaced, and the nested `keypoints`, `model`,
`postprocess` merge key by key.

| key | default | meaning |
|---|---|---|
| `name` | required | label for the project |
| `data_root` | required | folder holding one subfolder per subject (read only) |
| `subjects` | `[]` | subject folders, in order (the order fixes training-row order); empty = `data_root` itself is one subject |
| `video_include` | `["*.avi"]` | file-name patterns that are videos |
| `video_exclude` | `[]` | patterns to skip, e.g. `["*_concat.avi"]` |
| `dlc_tag` | required | text identifying the DLC network in the csv name |
| `dlc_individual` | `null` | multi-animal DLC: the individual to score |
| `annotation_names` | `["{stem}.avi.tsv"]` | BORIS export names to look for, first match wins |
| `behaviors` | `[]` | extra BORIS behaviours to store as labels in the cache |
| `behavior` | `approach` | the behaviour a model learns and predicts |
| `output_dir` | required | where `features/`, `models/`, `predictions/` go |
| `fps_default` | `30.0` | frame rate when a video has no BORIS export |
| `keypoints.snout`, `.neck`, `.center`, `.tailbase` | `snout`, `neck`, `spinecenter`, `tailbase` | DLC bodypart for each role: head axis = snout $-$ neck, body axis = neck $-$ tailbase, velocity from center |
| `keypoints.quality` | `[]` | bodyparts whose mean likelihood is the `mouse_L` feature (empty: no `mouse_L`) |
| `keypoints.target` | `[]` | candidate target bodyparts; each frame the most confident is the target (ties to the first). Empty: no target features |
| `keypoint_smoothing` | `5` | centred median on keypoint x and y, frames |
| `window_radii` | `[2, 5, 10, 20]` | window-feature radii, frames |
| `model_name` | `model` | name `train` saves under and `predict` loads |
| `drop_features` | `[]` | base features `train` leaves out, e.g. `[tgt_L, mouse_L]` |
| `model` | see Method | `HistGradientBoostingClassifier` arguments to override |
| `postprocess.threshold`, `.smooth`, `.fill_gap`, `.min_bout` | `0.5`, `5`, `5`, `8` | turning probabilities into bouts (see Method) |

## Outputs

```
<output_dir>/
  features/
    index.csv                    one row per video seen by `features`/`predict`
    <stem>.pkl                   per-frame features and labels for one video
  models/
    <name>.joblib                a trained model
    <name>/loao_scores.csv       leave-one-animal-out scores (train --cv)
    <name>/loao_predictions.pkl  leave-one-animal-out probabilities (train --cv)
  predictions/
    <model>/
      <stem>_<beh>_frames.csv    per-frame probability and call
      <stem>_<beh>_bouts.csv     one row per predicted bout
      <stem>_<beh>.tsv           the bouts as a BORIS tabular export
      summary.csv                one row per predicted video
```

### `features/index.csv`

| column | meaning |
|---|---|
| `subject`, `stem` | where the video came from |
| `avi`, `tsv`, `dlc` | paths of the video, BORIS export and DLC csv found (blank if none) |
| `n_frames`, `fps` | frames in the DLC csv; frame rate used |
| `status` | `scored`, `no_behavior`, `empty_tsv`, `no_tsv` or `no_dlc` (see Inputs) |
| `n_behavior` | number of `behavior` bouts in the export |
| `quirks` | oddities in the export, `;`-separated: `nan_image_index`, `<beh>:point_events`, `<beh>:unclosed_start`, `<beh>:orphan_stop`, `<beh>:unclosed_start_end` |

### `features/<stem>.pkl`

A pandas DataFrame, one row per frame: the per-frame features (see Method), one
boolean `y_<behaviour>` column per label when there is an export, and `frame`. Window
features are not cached; they are rebuilt from these when training or predicting.
Delete the folder, or pass `--overwrite`, after changing `keypoints`,
`keypoint_smoothing` or the DLC files.

### `models/<name>.joblib`

A dict (`joblib.load`): the fitted classifier (`model`), its input `columns`, the
post-processing (`threshold`, `smooth`, `fill_gap`, `min_bout`), `radii`, `behavior`,
`dropped` features, the videos and subjects it was `trained_on`, `n_frames` and
`n_pos`, the `dlc_tag`, `keypoints` and full `config`, and the scikit-learn version.
`predict` warns when the current config would compute features differently from the
one stored here.

### `models/<name>/loao_scores.csv` and `loao_predictions.pkl`

Each subject in turn is held out, a model is trained on the others, and the held-out
animal is scored. `loao_scores.csv` has one row per subject and one `ALL` row pooled
over every frame:

| column | meaning |
|---|---|
| `frames`, `pos_frac` | frames scored; fraction labelled as the behaviour |
| `AP` | average precision of the raw probability (threshold-free) |
| `precision`, `recall`, `F1` | per frame, after post-processing |
| `gt_bouts`, `bout_recall` | manual bouts; fraction overlapped by at least one predicted bout |

`loao_predictions.pkl` is the row-level table behind it: `subject`, `stem`, `frame`,
`y` (manual label) and `p` (held-out probability).

### `predictions/<model>/`

| file | columns |
|---|---|
| `<stem>_<beh>_frames.csv` | `frame`; `time_s`; `p`, the classifier's probability; `p_smooth`, after the running mean; `<beh>`, the final 0/1 call after thresholding, gap filling and minimum bout length |
| `<stem>_<beh>_bouts.csv` | `bout`, `start_frame`, `stop_frame` (exclusive), `n_frames`, `start_s`, `stop_s`, `mean_p` |
| `<stem>_<beh>.tsv` | the same bouts as BORIS `START`/`STOP` rows (`Description` = pyJAABA, `Comment` = auto); it reads back exactly through pyJAABA's own BORIS parser |
| `summary.csv` | `subject`, `stem`, `frames`, `pct_behavior`, `bouts`, `in_training_set` (a prediction on a training video is not a test). Rows from earlier runs into the same folder are kept |

## Method

1. **Per-frame features.** Keypoints get a centred median (`keypoint_smoothing`).
   Then, per frame (pixels, radians, per second):

   | feature | definition |
   |---|---|
   | `speed`, `accel` | speed of `center`, and its rate of change |
   | `dtheta_body`, `dtheta_head` | turn rate of the body axis (neck $-$ tailbase) and head axis (snout $-$ neck) |
   | `head_body_angle` | head axis relative to body axis |
   | `body_len`, `snout_neck_len` | lengths of the body and head axes |
   | `v_forward`, `v_lateral` | velocity along the body axis; magnitude across it |
   | `dist`, `log_dist`, `ddist` | snout-to-target distance, $\log(1 + d)$, and its rate |
   | `abs_Phi`, `cos_Phi`, `abs_Phi_head` | target bearing relative to the body axis ($\lvert\Phi\rvert$, $\cos\Phi$) and head axis |
   | `dalpha` | turn rate of the snout-to-target line |
   | `v_toward_tgt` | velocity of `center` toward the target |
   | `tgt_speed`, `rel_speed` | target speed; speed of the target relative to the mouse |
   | `mouse_L`, `tgt_L` | mean DLC likelihood of the `quality` bodyparts; likelihood of the chosen target |
   | `cx`, `cy` | arena position of `center` |

   Target rows are omitted when `keypoints.target` is empty; `mouse_L` when
   `keypoints.quality` is. Because units are pixels, a model is tied to the camera
   geometry it was trained on.
2. **Window features.** Each feature except arena position gets a centred mean, std,
   min and max at each radius $r$, plus the change $x_{t+r} - x_{t-r}$: 443 columns with
   a target and the default radii.
3. **Classifier.** `HistGradientBoostingClassifier` (300 iterations, learning rate 0.1,
   31 leaves, class-balanced, `random_state=0`; overridable under `model:`), trained on
   every frame of every trainable video.
4. **Post-processing.** Centred running mean of $p$ over `smooth` frames, threshold,
   fill gaps shorter than `fill_gap`, drop bouts shorter than `min_bout`. This barely
   changes frame-level F1 but merges the frame-by-frame flicker into bouts shaped like
   manual ones.

`drop_features: [tgt_L, mouse_L]` is recommended for new data: likelihoods measure
tracking quality, not behaviour, so a model leaning on them can drift with a new
network or recording setup.

## Tests

```bash
python tests/test_generic.py    # synthetic project: renamed bodyparts, multi-animal csv, no-target, train/CV/predict
```

`verify` is the real-data check: it compares a config's cached features (and with
`--loao`, its leave-one-animal-out predictions) against a reference folder holding
`frames.pkl`, `winfeat.pkl` and `pred_<behavior>_hgb.pkl` from an earlier run. On the
validation project above it reproduces them exactly ($\max|\Delta p| = 0$).
