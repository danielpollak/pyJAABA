"""Video discovery, BORIS label parsing, per-frame pose features and JAABA-style window features.

The feature arithmetic is unchanged from the investigation that validated the model,
with the bodypart names lifted into the config: with that project's config, `verify`
reproduces the reported features and leave-one-animal-out predictions exactly.

Units are DLC **pixels** and radians; nothing is converted to physical units. A model
is therefore tied to the camera geometry it was trained on.
"""
from __future__ import annotations

import fnmatch
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, Keypoints

# A video is usable for training only if every behaviour was scored exhaustively:
# an empty export means "watched, nothing happened", which is a valid all-negative video.
TRAINABLE = ("scored", "no_behavior", "empty_tsv")
# Absolute arena position: used raw, never windowed.
NO_WINDOW = ("cx", "cy")
# Features that need a target; omitted when the config names none.
TARGET_FEATURES = ("dist", "log_dist", "ddist", "abs_Phi", "cos_Phi", "abs_Phi_head", "dalpha",
                   "v_toward_tgt", "tgt_speed", "rel_speed", "tgt_L")


@dataclass
class Video:
    subject: str
    stem: str
    avi: str | None
    tsv: str | None
    dlc: str | None


# ------------------------------------------------------------------ discovery
def _match(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def find_tsv(stem, files, cfg: Config, subject: str = ""):
    """The video's BORIS export: `annotation_names` are file-name patterns in which
    {stem} and {subject} are substituted and * / ? are wildcards."""
    pats = [n.format(stem=glob.escape(stem), subject=glob.escape(subject)) for n in cfg.annotation_names]
    c = [f for f in files if _match(os.path.basename(f), pats)]
    if len(c) > 1:
        raise RuntimeError(f"more than one BORIS export for {stem}: {c}")
    return c[0] if c else None


def find_dlc(stem, files, cfg: Config):
    c = [f for f in files if os.path.basename(f).startswith(stem) and cfg.dlc_tag in f and f.endswith(".csv")]
    if len(c) > 1:
        raise RuntimeError(f"more than one {cfg.dlc_tag} DLC csv for {stem}: {c}")
    return c[0] if c else None


def discover(cfg: Config, subjects=None, root=None) -> list[Video]:
    """Every matching video under <root>/<subject>/, with its BORIS export and DLC csv.

    With no subjects, <root> itself is scanned and its folder name is the subject."""
    root = str(root or cfg.data_root)
    subjects = cfg.subjects if subjects is None else subjects
    dirs = [(s, os.path.join(root, s)) for s in subjects] or [(os.path.basename(root.rstrip("\\/")), root)]
    out = []
    for subj, d in dirs:
        files = glob.glob(os.path.join(d, "*"))
        vids = sorted(f for f in files if _match(os.path.basename(f), cfg.video_include)
                      and not _match(os.path.basename(f), cfg.video_exclude))
        for v in vids:
            stem = os.path.splitext(os.path.basename(v))[0]
            out.append(Video(subj, stem, v, find_tsv(stem, files, cfg, subj), find_dlc(stem, files, cfg)))
    return out


# ------------------------------------------------------------------ BORIS
def parse_tsv(path):
    """BORIS tabular events export -> ([(behavior, start, stop)], quirks, fps).

    start/stop are BORIS 'Image index' frames; the label covers frames [start, stop)."""
    df = pd.read_csv(path, sep="\t")
    rows, quirks = [], []
    if len(df) == 0:
        return rows, quirks, None
    fps = float(df["FPS"].iloc[0])
    # NaN Image index -> next row's value (else the timestamp)
    idx = df["Image index"].astype(float).values
    for i in range(len(idx)):
        if np.isnan(idx[i]):
            quirks.append("nan_image_index")
            j = i + 1
            while j < len(idx) and np.isnan(idx[j]):
                j += 1
            idx[i] = idx[j] if j < len(idx) else np.round(df["Time"].iloc[i] * fps)
    df["frame"] = idx.astype(int)
    for b, g in df.groupby("Behavior"):
        g = g.sort_values(["Time"], kind="stable")
        if any(t not in ("START", "STOP") for t in g["Behavior type"]):
            quirks.append(f"{b}:point_events")
        st = None
        for _, r in g.iterrows():
            if r["Behavior type"] == "START":
                if st is not None:
                    quirks.append(f"{b}:unclosed_start")
                st = r["frame"]
            elif r["Behavior type"] == "STOP":
                if st is None:
                    quirks.append(f"{b}:orphan_stop")
                    continue
                rows.append((b, st, r["frame"]))
                st = None
        if st is not None:
            quirks.append(f"{b}:unclosed_start_end")
    return rows, quirks, fps


def status_of(rows, fps, behavior):
    if fps is None:
        return "empty_tsv"
    return "scored" if any(r[0] == behavior for r in rows) else "no_behavior"


def labels(rows, T, behaviors):
    lab = {b: np.zeros(T, bool) for b in behaviors}
    for b, a, z in rows:
        if b in lab:
            lab[b][max(a, 0):min(z, T)] = True
    return lab


# ------------------------------------------------------------------ DLC
def load_dlc(path, individual: str | None = None) -> pd.DataFrame:
    """DLC csv -> columns (bodypart, coord). Multi-animal files are reduced to one
    individual, plus the 'single' unique bodyparts (e.g. a prey item)."""
    with open(path) as fh:
        head = [next(fh) for _ in range(4)]
    if head[1].split(",")[0].strip() != "individuals":
        d = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        d.columns = d.columns.droplevel(0)
        return d
    d = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
    d.columns = d.columns.droplevel(0)
    inds = [i for i in dict.fromkeys(d.columns.get_level_values(0)) if i != "single"]
    if individual is None:
        if len(inds) != 1:
            raise ValueError(f"{path}: multi-animal DLC with individuals {inds}; set dlc_individual")
        individual = inds[0]
    if individual not in inds:
        raise ValueError(f"{path}: no individual {individual!r}; have {inds}")
    parts = [d[individual]]
    if "single" in d.columns.get_level_values(0):
        s = d["single"]
        parts.append(s.loc[:, [c for c in s.columns if c[0] not in parts[0].columns.get_level_values(0)]])
    return pd.concat(parts, axis=1)


def check_keypoints(d: pd.DataFrame, kp: Keypoints, path=""):
    have = set(d.columns.get_level_values(0))
    need = [kp.snout, kp.neck, kp.center, kp.tailbase, *kp.quality, *kp.target]
    miss = [k for k in dict.fromkeys(need) if k not in have]
    if miss:
        raise ValueError(f"{path}: bodypart(s) {miss} not in DLC output; have {sorted(have)}")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def frame_features(d: pd.DataFrame, fps: float, kp: Keypoints, smooth: int = 5) -> pd.DataFrame:
    """Per-frame pursuit features from a single-animal DLC table (pixels; angles radians)."""
    xy = lambda k: d[k]["x"].values + 1j * d[k]["y"].values  # noqa: E731
    has_tgt = bool(kp.target)
    if has_tgt:
        tgt = xy(kp.target[0])
        tL = d[kp.target[0]]["likelihood"].values
        for k in kp.target[1:]:            # most confident candidate, ties to the earlier one
            Lk = d[k]["likelihood"].values
            tgt = np.where(Lk > tL, xy(k), tgt)
            tL = np.maximum(Lk, tL)

    def sm(z):  # centred median, applied to x and y separately
        return pd.Series(z).rolling(smooth, center=True, min_periods=1).median().values if np.isrealobj(z) else \
            sm(z.real) + 1j * sm(z.imag)

    c = sm(xy(kp.center)); s = sm(xy(kp.snout)); n = sm(xy(kp.neck)); tb = sm(xy(kp.tailbase))
    dt = 1.0 / fps
    vel = np.gradient(c) / dt
    body = n - tb
    head = s - n
    th_b = np.angle(body); th_h = np.angle(head)
    cols = {
        "speed": np.abs(vel),
        "accel": np.gradient(np.abs(vel)) / dt,
        "dtheta_body": wrap(np.gradient(np.unwrap(th_b))) / dt,
        "dtheta_head": wrap(np.gradient(np.unwrap(th_h))) / dt,
        "head_body_angle": wrap(th_h - th_b),
        "body_len": np.abs(body),
        "snout_neck_len": np.abs(head),
    }
    if has_tgt:
        t = sm(tgt)
        tvel = np.gradient(t) / dt
        los = t - s
        alpha = np.angle(los)
        Phi = wrap(alpha - th_b)
        Phi_h = wrap(alpha - th_h)
        dist = np.abs(los)
        cols |= {
            "dist": dist,
            "log_dist": np.log1p(dist),
            "ddist": np.gradient(dist) / dt,
            "abs_Phi": np.abs(Phi),
            "cos_Phi": np.cos(Phi),
            "abs_Phi_head": np.abs(Phi_h),
            "dalpha": np.abs(wrap(np.gradient(np.unwrap(alpha)))) / dt,
            "v_toward_tgt": (vel * np.conj(los / np.maximum(dist, 1e-6))).real,
        }
    cols |= {
        "v_lateral": np.abs((vel * np.conj(np.exp(1j * th_b))).imag),
        "v_forward": (vel * np.conj(np.exp(1j * th_b))).real,
    }
    if has_tgt:
        cols |= {"tgt_speed": np.abs(tvel), "rel_speed": np.abs(tvel - vel)}
    if kp.quality:
        cols["mouse_L"] = np.mean([d[k]["likelihood"].values for k in kp.quality], axis=0)
    if has_tgt:
        cols["tgt_L"] = tL
    cols |= {"cx": c.real, "cy": c.imag}
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)


def window_features(base: pd.DataFrame, radii) -> pd.DataFrame:
    """JAABA-style window features for ONE video: raw columns, then for each radius r the
    centred (2r+1)-frame mean/std/min/max and the change x[t+r] - x[t-r]."""
    cols = list(base.columns)
    win = [c for c in cols if c not in NO_WINDOW]
    df = pd.DataFrame({c: base[c].values.astype(np.float32) for c in cols}, index=base.index)
    out = [df]
    for r in radii:
        roll = df[win].rolling(2 * r + 1, center=True, min_periods=1)
        ch = df[win].shift(-r) - df[win].shift(r)
        for name, x in (("mean", roll.mean()), ("std", roll.std()), ("min", roll.min()),
                        ("max", roll.max()), ("chg", ch)):
            out.append(pd.DataFrame(x).add_suffix(f"_{name}{r}"))
    return pd.concat(out, axis=1).astype(np.float32)
