#!/usr/bin/env python3
"""
train_yolov8.py
- Single entrypoint to train YOLOv8 with small-object (basketball) friendly defaults.
- Uses inline defaults, but can optionally merge a YAML config via --cfg.
- Any CLI flag overrides both defaults and YAML values.

Usage
-----
# 1) Run with built-in defaults (recommended starting point)
python src/yolo/train_yolov8.py

# 2) Use a YAML of overrides
python src/yolo/train_yolov8.py --cfg configs/train_smallobj.yaml

# 3) Override individual params from CLI
python src/yolo/train_yolov8.py --epochs 80 --imgsz 960 --batch 8 --device 0

Notes
-----
- No 'hyp=' arg is used; we pass all hyperparameters directly as YOLO.train kwargs.
- Keep val set frozen in configs/dataset.yaml (val: images/val_fixed).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    import yaml  # optional, only needed if --cfg is used
except Exception:
    yaml = None

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from ultralytics import YOLO, settings as ysettings


def load_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is not installed. `pip install pyyaml` or omit --cfg.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"--cfg not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"--cfg must be a mapping, got: {type(data)}")
    return data


def merged_train_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build final training kwargs: defaults -> YAML (optional) -> CLI overrides."""
    # 1) Small-object friendly defaults (good starting point for ball)
    cfg: Dict[str, Any] = dict(
        # --- Core task/run settings ---
        task="detect",
        model="artifacts/weights/yolov8x.pt",                 # change to yolov8l.pt if GPU is tight
        data="configs/dataset.yaml",
        epochs=100,
        imgsz=1280,
        batch=8,
        device=0,                           # set None to auto
        project="runs/detect",
        name="y8x_smallobj",
        seed=0,
        deterministic=True,
        workers=8,

        # --- Optimization/regularization ---
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # --- Augmentations (geometry kept gentle for tiny ball) ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        shear=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        erasing=0.4,

        # --- Mosaic schedule (on early, off late) ---
        mosaic=1.0,
        close_mosaic=10,

        # --- Loss balance (tight boxes matter for tiny objects) ---
        box=7.5,
        cls=0.5,
        dfl=1.5,
        amp=False,
    )

    # 2) Merge YAML (if any)
    y = {}
    if args.cfg:
        y = load_yaml(args.cfg)
        # Filter out None / empty values
        y = {k: v for k, v in y.items() if v is not None}
        cfg.update(y)

    # 3) Apply CLI overrides (only those explicitly provided)
    cli_overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "workers": args.workers,
        # hyperparams (allow overriding main ones from CLI if desired)
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "shear": args.shear,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "erasing": args.erasing,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "amp": (args.amp == "true"), 
    }
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v

    # Remove keys Ultralytics doesn't expect at train-time
    # (we keep 'task' harmlessly; YOLO ignores unknown keys)
    return cfg



def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--cfg", type=str, default=None, help="Optional YAML of overrides")
    ap.add_argument("--model", type=str, default=None, help="e.g., yolov8x.pt or path/to/best.pt")
    ap.add_argument("--data", type=str, default=None, help="configs/dataset.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="0,1,.. or 'cpu'")
    ap.add_argument("--project", type=str, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--amp", type=str, default="false", choices=["true","false"],
                help="Enable mixed precision (may trigger internet AMP check).")


    # Allow overriding a few key hyperparams from CLI if needed
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--lrf", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--warmup_epochs", type=float, default=None)
    ap.add_argument("--warmup_momentum", type=float, default=None)
    ap.add_argument("--hsv_h", type=float, default=None)
    ap.add_argument("--hsv_s", type=float, default=None)
    ap.add_argument("--hsv_v", type=float, default=None)
    ap.add_argument("--degrees", type=float, default=None)
    ap.add_argument("--shear", type=float, default=None)
    ap.add_argument("--translate", type=float, default=None)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--fliplr", type=float, default=None)
    ap.add_argument("--flipud", type=float, default=None)
    ap.add_argument("--erasing", type=float, default=None)
    ap.add_argument("--mosaic", type=float, default=None)
    ap.add_argument("--close_mosaic", type=int, default=None)
    ap.add_argument("--box", type=float, default=None)
    ap.add_argument("--cls", type=float, default=None)
    ap.add_argument("--dfl", type=float, default=None)

    args = ap.parse_args()

    # Optional: ensure Ultralytics writes runs inside repo "runs"
    repo_runs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
    try:
        ysettings.update({"runs_dir": repo_runs})
    except Exception:
        pass

    train_kwargs = merged_train_args(args)


    # Sanity: show final args
    print("\n=== YOLOv8 train() kwargs ===")
    print(json.dumps(train_kwargs, indent=2, sort_keys=True))

    # Basic checks
    data_yaml = train_kwargs.get("data")
    if not os.path.exists(data_yaml):
        print(f"\n[WARN] dataset.yaml missing at: {data_yaml}", file=sys.stderr)

    # Train
    model_path = train_kwargs.pop("model", "yolov8x.pt")
    model = YOLO(model_path)

    print("\n[INFO] Starting training...")
    results = model.train(**train_kwargs)

    # Report where artifacts live
    out_project = train_kwargs.get("project", "runs/detect")
    out_name = train_kwargs.get("name", "exp")
    out_dir = os.path.join(out_project, out_name)
    print(f"\n=== Training complete ===\nOutputs: {out_dir}")
    print("Check results.png, confusion_matrix*.png, and weights/best.pt")


if __name__ == "__main__":
    main()