#!/usr/bin/env python3
"""
eval_yolov8.py
- Evaluate a trained YOLOv8 model on the dataset specified in dataset.yaml.
- Saves PR curves, confusion matrices, etc., under runs/detect/<name>.

Example
-------
python src/yolo/eval_yolov8.py \
  --model runs/detect/y8x_smallobj/weights/best.pt \
  --data configs/dataset.yaml \
  --imgsz 1280 --device 0 \
  --project runs/val --name y8x_eval
"""
import argparse
import os
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained weights .pt")
    ap.add_argument("--data", required=True, help="Path to YOLO dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/val")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--exist-ok", action="store_true")
    # note: Ultralytics 'val' uses split defined in YAML as 'val'
    args = ap.parse_args()

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        plots=True,   # saves curves and confusion matrices
        verbose=True,
    )

    out_dir = os.path.join(args.project, args.name)
    print("\n=== Validation complete ===")
    print(f"Outputs: {out_dir}")

    # Print a concise summary of key metrics
    # Keys vary by version; use getattr with defaults.
    def g(k, dflt=None): return metrics.results_dict.get(k, dflt)

    print("\n--- Key metrics ---")
    print(f"box/mAP50-95: {g('metrics/mAP50-95(B)')}")
    print(f"box/mAP50:    {g('metrics/mAP50(B)')}")
    print(f"Precision:    {g('metrics/precision(B)')}")
    print(f"Recall:       {g('metrics/recall(B)')}")
    print(f"Confusion matrices and PR curves saved to: {out_dir}")


if __name__ == "__main__":
    main()