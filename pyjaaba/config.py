"""Project configuration: one YAML describes where the videos are, how the DLC
keypoints map onto the roles the features need, and the model and post-processing.

A config may name a parent with ``extends: other.yaml`` (relative to itself). The
child's top-level keys replace the parent's; nested mappings (``keypoints``,
``model``, ``postprocess``) are merged key by key. Relative paths are resolved
against the YAML file that sets them. Unknown keys are an error, so a typo cannot
silently fall back to a default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

_NESTED = ("keypoints", "model", "postprocess")
_PATHS = ("data_root", "output_dir")


@dataclass
class Keypoints:
    """DLC bodypart names for each role the features use.

    snout, neck, center, tailbase  required; body axis = neck - tailbase, head axis
                                   = snout - neck, velocity from center.
    quality                        averaged into the `mouse_L` likelihood feature.
    target                         optional. Candidate target bodyparts; on each frame
                                   the one DLC is most confident about is the target
                                   (ties go to the first listed). Empty = no target,
                                   and every target-relative feature is omitted.
    """
    snout: str = "snout"
    neck: str = "neck"
    center: str = "spinecenter"
    tailbase: str = "tailbase"
    quality: list[str] = field(default_factory=list)
    target: list[str] = field(default_factory=list)


@dataclass
class Postprocess:
    threshold: float = 0.5
    smooth: int = 5      # centred running mean on p before thresholding (frames)
    fill_gap: int = 5    # merge bouts separated by fewer than this many frames
    min_bout: int = 8    # then drop bouts shorter than this


@dataclass
class Config:
    name: str
    data_root: Path
    output_dir: Path
    dlc_tag: str
    subjects: list[str] = field(default_factory=list)     # order fixes training-row order
    video_include: list[str] = field(default_factory=lambda: ["*.avi"])
    video_exclude: list[str] = field(default_factory=list)
    annotation_names: list[str] = field(default_factory=lambda: ["{stem}.avi.tsv"])
    dlc_individual: str | None = None                     # multi-animal DLC only
    fps_default: float = 30.0
    behaviors: list[str] = field(default_factory=list)    # BORIS behaviours to read as labels
    behavior: str = "approach"                            # the one a model scores
    model_name: str = "model"
    drop_features: list[str] = field(default_factory=list)
    keypoints: Keypoints = field(default_factory=Keypoints)
    keypoint_smoothing: int = 5
    window_radii: list[int] = field(default_factory=lambda: [2, 5, 10, 20])
    model: dict = field(default_factory=dict)             # HistGradientBoostingClassifier kwargs
    postprocess: Postprocess = field(default_factory=Postprocess)
    source: Path | None = None                            # the YAML this came from

    # derived locations
    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    def label_names(self) -> list[str]:
        return list(dict.fromkeys([self.behavior, *self.behaviors]))

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in (*_PATHS, "source"):
            d[k] = None if d[k] is None else str(d[k])
        return d

    @classmethod
    def load(cls, path) -> "Config":
        raw = _read(Path(path), set())
        return cls.from_dict(raw, source=Path(path).resolve())

    @classmethod
    def from_dict(cls, raw: dict, source: Path | None = None) -> "Config":
        known = {f.name for f in fields(cls)} - {"source"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config key(s) {sorted(unknown)}; allowed: {sorted(known)}")
        missing = {"name", "data_root", "output_dir", "dlc_tag"} - set(raw)
        if missing:
            raise ValueError(f"config is missing required key(s) {sorted(missing)}")
        raw = dict(raw)
        kp = raw.pop("keypoints", {}) or {}
        bad = set(kp) - {f.name for f in fields(Keypoints)}
        if bad:
            raise ValueError(f"unknown keypoints role(s) {sorted(bad)}")
        pp = raw.pop("postprocess", {}) or {}
        bad = set(pp) - {f.name for f in fields(Postprocess)}
        if bad:
            raise ValueError(f"unknown postprocess key(s) {sorted(bad)}")
        for k in ("subjects", "video_include", "video_exclude", "annotation_names", "behaviors",
                  "drop_features", "window_radii"):
            if k in raw and raw[k] is None:      # a YAML list with every entry commented out
                raw[k] = []
        for k in _PATHS:
            raw[k] = Path(raw[k])
        return cls(**raw, keypoints=Keypoints(**kp), postprocess=Postprocess(**pp), source=source)


def _read(path: Path, seen: set) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular `extends` at {path}")
    seen.add(path)
    with open(path, encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    for k in _PATHS:            # resolve relative paths against the file that sets them
        if k in d and d[k] is not None and not _is_abs(d[k]):
            d[k] = str((path.parent / d[k]).resolve())
    parent = d.pop("extends", None)
    if parent is None:
        return d
    base = _read(path.parent / parent, seen)
    for k, v in d.items():
        if k in _NESTED and isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def _is_abs(p: str) -> bool:
    return str(p).startswith(("\\\\", "//")) or Path(p).is_absolute()
