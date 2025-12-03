#!/usr/bin/env python3
"""
to_mot_format.py
- Convert Ultralytics 'yolo track --save-txt' outputs into MOTChallenge .txt files.

Assumptions
-----------
Ultralytics tracking txt lines are usually either:
  A) (pixels, left-top xywh)        -> already MOT-friendly
  B) (center-normalized xywh [0..1])-> need video W,H to convert

We auto-detect normalization; you can override with --norm if needed.

Examples
--------
python src/track/to_mot_format.py \
  --track-dir runs/track/byte \
  --video-dir clips \
  --outdir runs/track/byte_mot

The script expects per-video txt files inside --track-dir (Ultralytics layout).
"""
import argparse
import glob
import os
from typing import Tuple, Optional
import cv2
import numpy as np


def video_size(path: str) -> Optional[Tuple[int, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (w, h)


def detect_normalized(x, y, w, h) -> bool:
    vals = [x, y, w, h]
    return all(0.0 <= v <= 1.0 for v in vals)


def center_to_ltwh(cx, cy, w, h):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    return x1, y1, w, h


def process_one_txt(txt_path: str, video_path_hint: str, out_path: str, force_norm: Optional[bool]):
    # Attempt to find the matching video to read W,H if needed
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    # Heuristics to find a video with same stem
    candidate = None
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = os.path.join(video_path_hint, stem + ext)
        if os.path.exists(p):
            candidate = p
            break
    W = H = None
    if candidate:
        sz = video_size(candidate)
        if sz:
            W, H = sz

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_lines = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try to parse common Ultralytics formats.
            # We accept: frame, id, x, y, w, h, conf, (class?)
            parts = line.split(",")
            if len(parts) < 7:
                # Some versions write space-separated; try again.
                parts = line.split()
            if len(parts) < 7:
                # Not a recognized line; skip.
                continue

            try:
                frame = int(float(parts[0]))
                tid   = int(float(parts[1]))
                x     = float(parts[2])
                y     = float(parts[3])
                w     = float(parts[4])
                h     = float(parts[5])
                conf  = float(parts[6])
            except Exception:
                continue

            is_norm = force_norm if force_norm is not None else detect_normalized(x, y, w, h)

            if is_norm:
                if W is None or H is None:
                    raise RuntimeError(
                        f"Normalized coords detected in {txt_path}, but video size unknown. "
                        f"Place the matching video under --video-dir with name '{stem}.*'."
                    )
                # assume center-normalized (common in YOLO txt); convert to pixel left-top
                cx, cy = x * W, y * H
                pw, ph = w * W, h * H
                x1, y1, pw, ph = center_to_ltwh(cx, cy, pw, ph)
            else:
                # assume already pixel left-top xywh (common in Ultralytics track txt)
                x1, y1, pw, ph = x, y, w, h

            # MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
            out_lines.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{pw:.2f},{ph:.2f},{conf:.4f},-1,-1,-1")

    with open(out_path, "w") as g:
        g.write("\n".join(out_lines))

    return len(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", required=True,
                    help="Folder containing per-video tracking txt files from `yolo track --save-txt`")
    ap.add_argument("--video-dir", required=True,
                    help="Folder containing the original videos (to read width/height if needed)")
    ap.add_argument("--outdir", required=True, help="Output folder for MOT .txt files")
    ap.add_argument("--norm", choices=["auto", "true", "false"], default="auto",
                    help="Treat coords as normalized center-xywh (true/false) or auto-detect")
    args = ap.parse_args()

    force_norm = None
    if args.norm == "true":
        force_norm = True
    elif args.norm == "false":
        force_norm = False

    os.makedirs(args.outdir, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(args.track-dir, "**", "*.txt"), recursive=True))
    if not txts:
        raise SystemExit(f"No .txt files found under {args.track_dir}")

    total = 0
    for txt_path in txts:
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        out_path = os.path.join(args.outdir, f"{stem}.txt")
        n = process_one_txt(txt_path, args.video_dir, out_path, force_norm)
        print(f"Wrote {out_path} ({n} lines)")
        total += n

    print(f"\n=== Done. Total lines: {total} ===")


if __name__ == "__main__":
    main()