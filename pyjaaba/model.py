"""Feature cache, training (with optional leave-one-animal-out CV), prediction and writers."""
from __future__ import annotations

import datetime as _dt
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from . import features as F
from .config import Config, Postprocess

# The hyperparameters validated on the original prey-capture data; a config's `model:` overrides any of them.
MODEL_DEFAULTS = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, class_weight="balanced",
                      random_state=0)


# ---------------------------------------------------------------- feature cache
def cache_video(v: F.Video, cfg: Config, overwrite=False) -> dict:
    """Per-frame features (+ labels when a BORIS export exists) for one video -> <cache>/<stem>.pkl."""
    rec = dict(subject=v.subject, stem=v.stem, avi=v.avi, tsv=v.tsv, dlc=v.dlc, n_frames=np.nan,
               fps=np.nan, status="no_tsv" if v.tsv is None else "", quirks="", n_behavior=np.nan)
    if v.dlc is None:
        rec["status"] = "no_dlc"
        return rec
    rows, quirks, fps = ([], [], None) if v.tsv is None else F.parse_tsv(v.tsv)
    if v.tsv is not None:
        rec["status"] = F.status_of(rows, fps, cfg.behavior)
        rec["n_behavior"] = sum(r[0] == cfg.behavior for r in rows)
    rec["quirks"] = ";".join(sorted(set(quirks)))
    rec["fps"] = fps if fps else cfg.fps_default
    path = cfg.cache_dir / f"{v.stem}.pkl"
    if path.exists() and not overwrite:
        rec["n_frames"] = len(pd.read_pickle(path))
        return rec
    d = F.load_dlc(v.dlc, cfg.dlc_individual)
    F.check_keypoints(d, cfg.keypoints, v.dlc)
    T = len(d)
    rec["n_frames"] = T
    f = F.frame_features(d, rec["fps"], cfg.keypoints, cfg.keypoint_smoothing)
    if v.tsv is not None:
        for b, y in F.labels(rows, T, cfg.label_names()).items():
            f["y_" + b] = y
    f["frame"] = np.arange(T)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    f.to_pickle(path)
    return rec


def build_cache(videos, cfg: Config, overwrite=False, log=print) -> pd.DataFrame:
    recs = []
    for i, v in enumerate(videos):
        r = cache_video(v, cfg, overwrite=overwrite)
        recs.append(r)
        log(f"[{i + 1}/{len(videos)}] {v.subject} {v.stem[:70]} {r['status']} {r['n_frames']}")
    idx = pd.DataFrame(recs)
    old = cfg.cache_dir / "index.csv"
    if old.exists():  # merge with videos cached earlier (e.g. other subjects)
        prev = pd.read_csv(old).rename(columns={"n_approach": "n_behavior"})
        idx = pd.concat([prev[~prev.stem.isin(idx.stem)], idx], ignore_index=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    idx.to_csv(old, index=False)
    return idx


def load_index(cfg: Config) -> pd.DataFrame:
    p = cfg.cache_dir / "index.csv"
    if not p.exists():
        raise FileNotFoundError(f"no feature cache at {cfg.cache_dir}; run the `features` command first")
    return pd.DataFrame(pd.read_csv(p))


def _base(f: pd.DataFrame) -> pd.DataFrame:
    return f[[c for c in f.columns if not c.startswith("y_") and c != "frame"]]


def training_table(cfg: Config, subjects=None):
    """Window features, labels and row metadata for every trainable cached video.

    Row order is subject order (as given), then video stem; the validated LOAO numbers
    depend on it because HGB's internal early-stopping split does."""
    subjects = cfg.subjects if subjects is None else subjects
    idx = load_index(cfg)
    idx = idx[idx.subject.isin(subjects) & idx.status.isin(F.TRAINABLE)].copy()
    if idx.empty:
        raise ValueError(f"no trainable (exhaustively scored) cached videos for {subjects}")
    idx["_s"] = idx.subject.map({s: i for i, s in enumerate(subjects)})
    idx = idx.sort_values(["_s", "stem"], kind="stable")
    Xs, meta = [], []
    for r in idx.itertuples():
        f = pd.read_pickle(cfg.cache_dir / f"{r.stem}.pkl")
        if "y_" + cfg.behavior not in f:
            raise ValueError(f"{r.stem}: cached without a '{cfg.behavior}' label; re-run features --overwrite")
        Xs.append(F.window_features(_base(f), cfg.window_radii))
        meta.append(pd.DataFrame({"subject": r.subject, "stem": r.stem, "frame": f.frame.values,
                                  "y": f["y_" + cfg.behavior].values.astype(int)}))
    return pd.concat(Xs, ignore_index=True), pd.concat(meta, ignore_index=True)


# ---------------------------------------------------------------- model
def make_model(cfg: Config):
    return HistGradientBoostingClassifier(**(MODEL_DEFAULTS | cfg.model))


_WIN = re.compile(r"^(.*)_(mean|std|min|max|chg)\d+$")


def base_of(col: str) -> str:
    m = _WIN.match(col)
    return m.group(1) if m else col


def select_columns(columns, drop=()):
    """Drop base features and all their window variants by name, e.g. drop=('tgt_L', 'mouse_L')."""
    bases = {base_of(c) for c in columns}
    unknown = set(drop) - bases
    if unknown:
        raise ValueError(f"unknown feature(s) to drop: {sorted(unknown)}; choose from {sorted(bases)}")
    return [c for c in columns if base_of(c) not in set(drop)]


def runs(b):
    d = np.diff(np.r_[0, np.asarray(b).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def postprocess(p, pp: Postprocess):
    """Probability -> (boolean per-frame calls, smoothed probability)."""
    ps = pd.Series(p).rolling(pp.smooth, center=True, min_periods=1).mean().to_numpy()
    b = ps >= pp.threshold
    for a, z in runs(~b):
        if a > 0 and z < len(b) and z - a < pp.fill_gap:
            b[a:z] = True
    for a, z in runs(b):
        if z - a < pp.min_bout:
            b[a:z] = False
    return b, ps


def scores(meta: pd.DataFrame, p: np.ndarray, pp: Postprocess) -> pd.DataFrame:
    """Frame P/R/F1 after post-processing, AP, and overlap bout recall, per subject and ALL."""
    m = meta.assign(p=p)
    rows = []
    for subj, gS in list(m.groupby("subject", sort=False)) + [("ALL", m)]:
        tp = fp = fn = nG = hit = 0
        for _, g in gS.groupby("stem", sort=False):
            gt = g.y.values.astype(bool)
            pr, _ = postprocess(g.p.values, pp)
            tp += (gt & pr).sum(); fp += (~gt & pr).sum(); fn += (gt & ~pr).sum()
            G, P = runs(gt), runs(pr)
            nG += len(G); hit += sum(any(pa < gz and ga < pz for pa, pz in P) for ga, gz in G)
        P_ = tp / max(tp + fp, 1); R_ = tp / max(tp + fn, 1)
        rows.append(dict(subject=subj, frames=len(gS), pos_frac=gS.y.mean(),
                         AP=average_precision_score(gS.y, gS.p) if gS.y.any() else np.nan,
                         precision=P_, recall=R_, F1=2 * P_ * R_ / max(P_ + R_, 1e-9),
                         gt_bouts=nG, bout_recall=hit / max(nG, 1)))
    return pd.DataFrame(rows)


def loao(X: pd.DataFrame, meta: pd.DataFrame, cfg: Config, log=print) -> np.ndarray:
    p = np.full(len(meta), np.nan)
    for S in sorted(meta.subject.unique()):
        te = (meta.subject == S).values
        t0 = time.time()
        m = make_model(cfg).fit(X.values[~te], meta.y.values[~te])
        p[te] = m.predict_proba(X.values[te])[:, 1]
        log(f"  held out {S}: {time.time() - t0:.0f}s")
    return p


def fit(X, meta, cfg: Config, drop=(), subjects=None) -> dict:
    cols = list(X.columns)
    m = make_model(cfg).fit(X.values, meta.y.values)
    pp = cfg.postprocess
    return dict(model=m, columns=cols, dropped=list(drop), behavior=cfg.behavior, radii=list(cfg.window_radii),
                threshold=pp.threshold, smooth=pp.smooth, fill_gap=pp.fill_gap, min_bout=pp.min_bout,
                trained_on=sorted(meta.stem.unique()), subjects=list(subjects or cfg.subjects),
                n_frames=len(meta), n_pos=int(meta.y.sum()), dlc_tag=cfg.dlc_tag,
                keypoints=cfg.to_dict()["keypoints"], keypoint_smoothing=cfg.keypoint_smoothing,
                config=cfg.to_dict(), sklearn=sklearn.__version__,
                created=_dt.datetime.now().isoformat(timespec="seconds"))


def save(art: dict, name: str, cfg: Config) -> Path:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.models_dir / f"{name}.joblib"
    joblib.dump(art, p)
    return p


def load(name_or_path, cfg: Config) -> dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = cfg.models_dir / f"{name_or_path}.joblib"
    art = joblib.load(p)
    if art["sklearn"] != sklearn.__version__:
        print(f"warning: model saved with scikit-learn {art['sklearn']}, running {sklearn.__version__}")
    return art


def check_compatible(art: dict, cfg: Config):
    """Warn when the config would compute features differently from the model's training run."""
    probs = []
    if art.get("dlc_tag") != cfg.dlc_tag:
        probs.append(f"DLC network {art.get('dlc_tag')!r} vs config {cfg.dlc_tag!r}")
    if "keypoints" in art and art["keypoints"] != cfg.to_dict()["keypoints"]:
        probs.append(f"keypoints {art['keypoints']} vs config {cfg.to_dict()['keypoints']}")
    if list(art.get("radii", cfg.window_radii)) != list(cfg.window_radii):
        probs.append(f"window radii {art.get('radii')} vs config {cfg.window_radii}")
    for p in probs:
        print(f"warning: model/config mismatch: {p}")


def postprocess_of(art: dict) -> Postprocess:
    return Postprocess(art["threshold"], art["smooth"], art["fill_gap"], art["min_bout"])


# ---------------------------------------------------------------- prediction
def predict_video(art: dict, base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = F.window_features(_base(base), art["radii"])
    missing = set(art["columns"]) - set(X.columns)
    if missing:
        raise ValueError(f"features missing for this model: {sorted(missing)[:5]}...")
    p = art["model"].predict_proba(X[art["columns"]].values)[:, 1]
    calls, ps = postprocess(p, postprocess_of(art))
    return p, ps, calls


BORIS_COLUMNS = ["Observation id", "Observation date", "Description", "Observation duration", "Observation type",
                 "Source", "Time offset (s)", "Media duration (s)", "FPS", "Subject", "Behavior",
                 "Behavioral category", "Behavior type", "Time", "Media file name", "Image index",
                 "Image file path", "Comment"]


def write_outputs(out_dir: Path, stem: str, fps: float, p, ps, calls, behavior, avi=None):
    """<stem>_<beh>_frames.csv, <stem>_<beh>_bouts.csv, and <stem>_<beh>.tsv (BORIS export layout)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n = len(p)
    pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / fps, "p": p.round(5),
                  "p_smooth": ps.round(5), behavior: calls.astype(int)}).to_csv(
        out_dir / f"{stem}_{behavior}_frames.csv", index=False)
    B = runs(calls)
    pd.DataFrame([dict(bout=i, start_frame=a, stop_frame=z, n_frames=z - a, start_s=a / fps, stop_s=z / fps,
                       mean_p=float(p[a:z].mean())) for i, (a, z) in enumerate(B)],
                 columns=["bout", "start_frame", "stop_frame", "n_frames", "start_s", "stop_s", "mean_p"]).to_csv(
        out_dir / f"{stem}_{behavior}_bouts.csv", index=False)
    # STOP is the first frame after the bout, matching parse_tsv, so parse_tsv(written)
    # recovers the bouts exactly.
    media = avi or (stem + ".avi")
    ev = []
    for a, z in B:
        for typ, fr in (("START", a), ("STOP", z)):
            ev.append({"Observation id": stem + ".avi", "Observation date": "", "Description": "pyJAABA",
                       "Observation duration": "", "Observation type": "Media file(s)", "Source": f"player #1:{media}",
                       "Time offset (s)": 0.0, "Media duration (s)": round(n / fps, 3), "FPS": round(fps, 3),
                       "Subject": "", "Behavior": behavior, "Behavioral category": "", "Behavior type": typ,
                       "Time": round(fr / fps, 3), "Media file name": media, "Image index": int(fr),
                       "Image file path": "NA", "Comment": "auto"})
    pd.DataFrame(ev, columns=BORIS_COLUMNS).to_csv(out_dir / f"{stem}_{behavior}.tsv", sep="\t", index=False)
    return B
