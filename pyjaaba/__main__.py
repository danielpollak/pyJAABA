"""pyJAABA: score a behaviour frame by frame from DeepLabCut pose.

    python -m pyjaaba info     --config project.yaml
    python -m pyjaaba features --config project.yaml [--subjects A B] [--dlc a.csv ...] [--overwrite]
    python -m pyjaaba train    --config project.yaml [--name NAME] [--cv] [--drop F ...]
    python -m pyjaaba predict  --config project.yaml [--model NAME|PATH] [--subjects A B | --dlc a.csv ...]
    python -m pyjaaba evaluate --config project.yaml [--model NAME|PATH]   # vs BORIS labels of unseen videos
    python -m pyjaaba verify   --config project.yaml --reference DIR [--loao]

Every default (subjects, model name, dropped features) comes from the config; a flag
overrides it for that run. See README.md.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from . import features as F
from . import model as M
from .config import Config


def _videos(args, cfg: Config) -> list:
    vids = []
    if args.subjects is None or args.subjects:
        vids += F.discover(cfg, args.subjects, args.root)
    for p in args.dlc or []:
        p = Path(p)
        stem = p.name.split("DLC_")[0] if "DLC_" in p.name else p.stem
        files = [str(c) for c in p.parent.iterdir()]
        avi = next((f for f in files if F._match(Path(f).name, cfg.video_include)
                    and Path(f).stem == stem), None)
        vids.append(F.Video(args.label or "unknown", stem, avi, F.find_tsv(stem, files, cfg, args.label or ""), str(p)))
    if not vids:
        sys.exit("nothing to do: no videos found (check data_root/subjects, or give --dlc)")
    return vids


def cmd_info(args, cfg):
    print(f"config  {cfg.source}")
    for k, v in cfg.to_dict().items():
        if k != "source":
            print(f"  {k}: {v}")
    vids = F.discover(cfg)
    if vids:
        df = pd.DataFrame([dict(subject=v.subject, video=1, dlc=v.dlc is not None, tsv=v.tsv is not None)
                           for v in vids])
        print("\nvideos found:")
        print(df.groupby("subject", sort=False).sum().to_string())
    else:
        print(f"\nno videos found under {cfg.data_root} for subjects {cfg.subjects}")


def cmd_features(args, cfg):
    idx = M.build_cache(_videos(args, cfg), cfg, overwrite=args.overwrite)
    print()
    print(idx.groupby(["subject", "status"]).size().unstack(fill_value=0).to_string())
    print(f"\ncache: {cfg.cache_dir}")


def cmd_train(args, cfg):
    subjects = cfg.subjects if args.subjects is None else args.subjects
    drop = cfg.drop_features if args.drop is None else args.drop
    name = args.name or cfg.model_name
    X, meta = M.training_table(cfg, subjects)
    cols = M.select_columns(list(X.columns), drop)
    X = X[cols]
    print(f"training table: {meta.stem.nunique()} videos, {len(meta)} frames, {int(meta.y.sum())} {cfg.behavior}, "
          f"{len(cols)} features" + (f" (dropped {', '.join(drop)})" if drop else ""))
    out = cfg.models_dir / name
    if args.cv:
        print("leave-one-animal-out:")
        p = M.loao(X, meta, cfg)
        s = M.scores(meta, p, cfg.postprocess)
        out.mkdir(parents=True, exist_ok=True)
        s.to_csv(out / "loao_scores.csv", index=False)
        meta.assign(p=p).to_pickle(out / "loao_predictions.pkl")
        pd.set_option("display.width", 200)
        print(s.round(3).to_string(index=False))
    art = M.fit(X, meta, cfg, drop=drop, subjects=subjects)
    print(f"saved {M.save(art, name, cfg)}")


def cmd_predict(args, cfg):
    model = args.model or cfg.model_name
    art = M.load(model, cfg)
    M.check_compatible(art, cfg)
    name = Path(model).stem
    out = Path(args.out) if args.out else cfg.predictions_dir / name
    rows = []
    for v in _videos(args, cfg):
        if v.dlc is None:
            print(f"skip {v.stem}: no {cfg.dlc_tag} DLC csv"); continue
        r = M.cache_video(v, cfg)
        base = pd.read_pickle(cfg.cache_dir / f"{v.stem}.pkl")
        p, ps, calls = M.predict_video(art, base)
        B = M.write_outputs(out, v.stem, r["fps"], p, ps, calls, art["behavior"], v.avi)
        seen = v.stem in art["trained_on"]
        rows.append(dict(subject=v.subject, stem=v.stem, frames=len(p), pct_behavior=100 * calls.mean(),
                         bouts=len(B), in_training_set=seen))
        print(f"{v.stem[:70]}: {len(B)} bouts, {100 * calls.mean():.1f}% {art['behavior']}" +
              ("  [WARNING: model was trained on this video]" if seen else ""))
    if rows:
        s = pd.DataFrame(rows)
        f = out / "summary.csv"
        if f.exists():        # keep rows from earlier runs on other videos
            prev = pd.read_csv(f).rename(columns={"pct_approach": "pct_behavior"})
            s = pd.concat([prev[~prev.stem.isin(s.stem)], s], ignore_index=True)
        s.to_csv(f, index=False)
        print(f"\nwrote {out}")


def cmd_evaluate(args, cfg):
    """Score a trained model against the BORIS labels of videos it was not trained on."""
    model = args.model or cfg.model_name
    art = M.load(model, cfg)
    M.check_compatible(art, cfg)
    name = Path(model).stem
    out = Path(args.out) if args.out else cfg.predictions_dir / name
    metas, ps = [], []
    for v in _videos(args, cfg):
        r = M.cache_video(v, cfg)
        if r["status"] not in F.TRAINABLE:
            print(f"skip {v.stem}: {r['status']} (needs a DLC csv and a complete BORIS export)"); continue
        if v.stem in art["trained_on"]:
            print(f"skip {v.stem}: the model was trained on it"); continue
        base = pd.read_pickle(cfg.cache_dir / f"{v.stem}.pkl")
        if "y_" + art["behavior"] not in base:
            print(f"skip {v.stem}: cached without a '{art['behavior']}' label; re-run features --overwrite"); continue
        p, ps_, calls = M.predict_video(art, base)
        M.write_outputs(out, v.stem, r["fps"], p, ps_, calls, art["behavior"], v.avi)
        metas.append(pd.DataFrame({"subject": v.subject, "stem": v.stem, "frame": base.frame.values,
                                   "y": base["y_" + art["behavior"]].values.astype(int)}))
        ps.append(p)
    if not metas:
        sys.exit("no labelled videos to evaluate")
    meta, p = pd.concat(metas, ignore_index=True), pd.concat([pd.Series(x) for x in ps]).to_numpy()
    s = M.scores(meta, p, M.postprocess_of(art))
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "evaluation.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
    print(f"\nwrote {out / 'evaluation.csv'} (+ predictions for each video)")


def cmd_verify(args, cfg):
    from .verify import verify
    verify(cfg, Path(args.reference), loao=args.loao)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pyjaaba", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--config", required=True, help="project YAML")

    def sources(p):
        p.add_argument("--subjects", nargs="*", default=None,
                       help="subject folders under data_root (default: the config's; give none to skip)")
        p.add_argument("--root", default=None, help="override data_root (read-only)")
        p.add_argument("--dlc", nargs="*", help="individual DLC csv files")
        p.add_argument("--label", help="subject label for --dlc files")

    pi = sub.add_parser("info", help="show the resolved config and the videos it finds")
    common(pi); pi.set_defaults(fn=cmd_info)

    pf = sub.add_parser("features", help="compute and cache per-frame features (+ BORIS labels)")
    common(pf); sources(pf)
    pf.add_argument("--overwrite", action="store_true")
    pf.set_defaults(fn=cmd_features)

    pt = sub.add_parser("train", help="fit on every exhaustively scored cached video; optionally LOAO-validate")
    common(pt)
    pt.add_argument("--name", help="model name (default: config model_name)")
    pt.add_argument("--subjects", nargs="*", default=None)
    pt.add_argument("--cv", action="store_true", help="also run leave-one-animal-out and report scores")
    pt.add_argument("--drop", nargs="*", default=None, help="base features to exclude (default: config drop_features)")
    pt.set_defaults(fn=cmd_train)

    pp = sub.add_parser("predict", help="write per-frame scores, bouts and a BORIS-style tsv per video")
    common(pp); sources(pp)
    pp.add_argument("--model", help="model name in <output_dir>/models, or a path (default: config model_name)")
    pp.add_argument("--out", help="output directory (default <output_dir>/predictions/<model>)")
    pp.set_defaults(fn=cmd_predict)

    pe = sub.add_parser("evaluate", help="score a model against BORIS labels of videos it was not trained on")
    common(pe); sources(pe)
    pe.add_argument("--model", help="model name in <output_dir>/models, or a path (default: config model_name)")
    pe.add_argument("--out", help="output directory (default <output_dir>/predictions/<model>)")
    pe.set_defaults(fn=cmd_evaluate)

    pv = sub.add_parser("verify", help="check features (and optionally LOAO) against reference arrays")
    common(pv)
    pv.add_argument("--reference", required=True, help="dir with frames.pkl, winfeat.pkl, pred_*_hgb.pkl")
    pv.add_argument("--loao", action="store_true")
    pv.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    if not hasattr(args, "dlc"):
        args.dlc = None
    args.fn(args, Config.load(args.config))


if __name__ == "__main__":
    main()
