"""Check that a config reproduces reference arrays exactly.

The reference directory holds what a validated run wrote:
  frames.pkl            per-frame features + labels (columns subject, stem, frame, status, y_<beh>, ...)
  winfeat.pkl           window features, row-aligned with frames.pkl
  pred_<beh>_hgb.pkl    leave-one-animal-out probabilities (column p), same rows (only for --loao)

Needs the feature cache (`features`) to exist for the config's subjects.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import model as M
from .config import Config


def _same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def verify(cfg: Config, ref: Path, loao=False):
    if not (ref / "frames.pkl").exists():
        print(f"SKIP: no reference arrays at {ref}")
        return
    fr = pd.read_pickle(ref / "frames.pkl")
    fr = fr[fr.status.isin(F.TRAINABLE)].reset_index(drop=True)
    X, meta = M.training_table(cfg)
    yname = "y_" + cfg.behavior

    assert (meta.stem.values == fr.stem.values).all(), "video set or row order differs"
    assert (meta.frame.values == fr.frame.values).all()
    assert (meta.y.values == fr[yname].values.astype(int)).all(), "labels differ"
    print(f"ok  rows/labels: {meta.stem.nunique()} videos, {len(meta)} frames, {meta.y.sum()} {cfg.behavior}")

    base = [c for c in fr.columns if not c.startswith("y_") and c not in ("subject", "stem", "frame", "status")]
    bad = [c for c in base if not _same(X[c].values, fr[c].values.astype(np.float32))]
    assert not bad, f"per-frame features differ: {bad}"
    print(f"ok  {len(base)} per-frame features identical")

    W = pd.read_pickle(ref / "winfeat.pkl").loc[fr.index]
    assert list(W.columns) == list(X.columns), "window-feature columns differ"
    bad = [c for c in W.columns if not _same(X[c].values, W[c].values)]
    assert not bad, f"window features differ: {bad[:5]}"
    print(f"ok  {W.shape[1]} window features identical")

    if loao:
        r = pd.read_pickle(ref / f"pred_{cfg.behavior}_hgb.pkl")
        p = M.loao(X, meta, cfg)
        err = float(np.max(np.abs(p - r.p.values)))
        s = M.scores(meta, p, cfg.postprocess).set_index("subject").loc["ALL"]
        print(f"LOAO: max |dp| vs reference {err:.1e}; AP {s.AP:.3f}, "
              f"P/R/F1 {s.precision:.3f}/{s.recall:.3f}/{s.F1:.3f}")
        assert err < 1e-9, "LOAO predictions differ from the reference run"
        print("ok  LOAO predictions reproduce the reference")
