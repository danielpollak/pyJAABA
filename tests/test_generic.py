"""Synthetic end-to-end run on a made-up DLC project: renamed bodyparts, a multi-animal
csv with a unique-bodypart target, a project with no target at all, and a BORIS export.
No real data needed.

    python tests/test_generic.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyjaaba import features as F  # noqa: E402
from pyjaaba import model as M  # noqa: E402
from pyjaaba.__main__ import main  # noqa: E402
from pyjaaba.config import Config  # noqa: E402

BODY = ["nose", "head", "mid", "tail0"]


def fake_dlc(path: Path, T: int, rng, multi=False, target="cricket"):
    """Mouse chases a target during frames [T/3, 2T/3)."""
    t = np.arange(T)
    tgt = np.c_[300 + 100 * np.cos(t / 40), 300 + 100 * np.sin(t / 40)]
    mouse = np.c_[100 + 0.2 * t, 100 + 0.1 * t]
    chase = (t >= T // 3) & (t < 2 * T // 3)
    for i in np.where(chase)[0]:
        mouse[i] = mouse[i - 1] + 0.15 * (tgt[i] - mouse[i - 1])
    hd = np.angle((tgt - mouse) @ [1, 1j])
    cols, data = [], []
    for k, off in zip(BODY, (12, 6, 0, -12)):
        xy = mouse + off * np.c_[np.cos(hd), np.sin(hd)] + rng.normal(0, 0.5, (T, 2))
        for c, v in zip(("x", "y", "likelihood"), (xy[:, 0], xy[:, 1], rng.uniform(.9, 1, T))):
            cols.append(("mouse1", k, c) if multi else (k, c)); data.append(v)
    if target:
        for c, v in zip(("x", "y", "likelihood"), (tgt[:, 0], tgt[:, 1], rng.uniform(.8, 1, T))):
            cols.append(("single", target, c) if multi else (target, c)); data.append(v)
    names = ["individuals", "bodyparts", "coords"] if multi else ["bodyparts", "coords"]
    df = pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols, names=names))
    df.columns = pd.MultiIndex.from_tuples([("DLC_fake",) + c for c in cols], names=["scorer"] + names)
    df.to_csv(path)
    return chase


def boris(path: Path, stem: str, chase: np.ndarray, fps=30.0):
    a, z = M.runs(chase)[0]
    rows = [{"Observation id": stem, "FPS": fps, "Behavior": "hunt", "Behavior type": k,
             "Time": f / fps, "Image index": f} for k, f in (("START", a), ("STOP", z))]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def main_test():
    rng = np.random.default_rng(0)
    root = Path(tempfile.mkdtemp(prefix="pyjaaba_test_"))
    for subj, multi in (("A", False), ("B", True), ("C", False)):
        d = root / "data" / subj
        d.mkdir(parents=True)
        for k in range(2):
            stem = f"{subj}_take{k}"
            (d / f"{stem}.avi").write_bytes(b"")
            chase = fake_dlc(d / f"{stem}DLC_fake_net.csv", 900, rng, multi=multi)
            boris(d / f"{stem}.avi.tsv", stem, chase)
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(f"""
name: synthetic
data_root: {(root / 'data').as_posix()}
subjects: [A, B, C]
dlc_tag: fake_net
behaviors: [hunt]
behavior: hunt
output_dir: out
keypoints: {{snout: nose, neck: head, center: mid, tailbase: tail0, quality: [nose, head], target: [cricket]}}
model_name: m
model: {{max_iter: 30}}
""")
    cfg = Config.load(cfg_path)
    assert cfg.output_dir == root / "out"
    main(["features", "--config", str(cfg_path)])
    main(["train", "--config", str(cfg_path), "--cv"])
    s = pd.read_csv(cfg.models_dir / "m" / "loao_scores.csv").set_index("subject")
    print(f"synthetic LOAO F1 {s.loc['ALL', 'F1']:.2f}")
    assert s.loc["ALL", "F1"] > 0.8, "classifier failed on an easy synthetic task"
    main(["predict", "--config", str(cfg_path), "--subjects", "C"])
    out = cfg.predictions_dir / "m"
    rows, _, _ = F.parse_tsv(out / "C_take0_hunt.tsv")
    assert rows and all(r[0] == "hunt" for r in rows)
    print("ok  single- and multi-animal DLC, renamed bodyparts, train/CV/predict, BORIS tsv round trip")

    # no target: target-relative features are omitted, the rest still works
    d = F.load_dlc(root / "data" / "A" / "A_take0DLC_fake_net.csv")
    f = F.frame_features(d, 30.0, cfg.keypoints.__class__(snout="nose", neck="head", center="mid",
                                                         tailbase="tail0"))
    assert not set(F.TARGET_FEATURES) & set(f.columns) and "speed" in f and "mouse_L" not in f
    print("ok  no-target project")

    # unknown keys and missing bodyparts fail loudly
    bad = root / "bad.yaml"
    bad.write_text(cfg_path.read_text() + "\nsubjcets: [A]\n")
    try:
        Config.load(bad); raise AssertionError("typo accepted")
    except ValueError:
        pass
    try:
        F.check_keypoints(d, cfg.keypoints.__class__(snout="nostril"), "x"); raise AssertionError
    except ValueError:
        pass
    print("ok  config typos and missing bodyparts rejected")


if __name__ == "__main__":
    main_test()
