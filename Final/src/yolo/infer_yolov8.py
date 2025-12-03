#!/usr/bin/env python3
"""
infer_yolov8.py
- Batch inference with Ultralytics YOLOv8 on images or videos.
- For CVAT prelabeling: run on frames with --save-txt to produce YOLO .txt files.

Examples
--------
# Prelabel frames for CVAT (writes YOLO .txt under runs/predict/<name>/labels)
python src/yolo/infer_yolov8.py \
  --model runs/detect/y8x_smallobj/weights/best.pt \
  --source "data/queue_for_labeling/**/*.jpg" \
  --imgsz 1280 --conf 0.20 --iou 0.6 --save-txt --save-conf \
  --project runs/predict --name prelabels

# Quick video demo overlay
python src/yolo/infer_yolov8.py \
  --model runs/detect/y8x_smallobj/weights/best.pt \
  --source clips/*.mp4 --save-img
"""
import argparse
import glob
import os
from typing import List

from ultralytics import YOLO


def expand_sources(patterns: List[str]) -> List[str]:
    out = []
    for p in patterns:
        # glob supports ** if recursive=True
        matches = glob.glob(p, recursive=True)
        if matches:
            out.extend(sorted(matches))
        elif os.path.exists(p):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pt (or .onnx, etc.)")
    ap.add_argument("--source", nargs="+", required=True,
                    help="Files/globs/dirs (images or videos). Use quotes for globs.")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default=None, help="CUDA device id or 'cpu' (e.g., 0)")
    ap.add_argument("--save-txt", action="store_true",
                    help="Save YOLO .txt predictions (useful for CVAT import).")
    ap.add_argument("--save-conf", action="store_true", help="Include confidences in txt.")
    ap.add_argument("--save-img", action="store_true",
                    help="Save rendered images/videos for visual check.")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="Filter by class id(s), e.g. 0 1")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--project", default="runs/predict")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--exist-ok", action="store_true")
    args = ap.parse_args()

    sources = expand_sources(args.source)
    if not sources:
        raise SystemExit(f"No inputs matched: {args.source}")

    model = YOLO(args.model)

    # Ultralytics predict() handles lists/dirs/globs transparently.
    results = model.predict(
        source=sources if len(sources) > 1 else sources[0],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=args.save_img,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        classes=args.classes,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
    )

    # Helpful summary
    out_dir = os.path.join(args.project, args.name)
    labels_dir = os.path.join(out_dir, "labels")
    print("\n=== Inference complete ===")
    print(f"Outputs: {out_dir}")
    if args.save_txt:
        print(f"YOLO txt predictions: {labels_dir} (one .txt per image/frame)")
    if args.save_img:
        print("Rendered previews saved under 'images/' inside the output folder.")


if __name__ == "__main__":
    main()