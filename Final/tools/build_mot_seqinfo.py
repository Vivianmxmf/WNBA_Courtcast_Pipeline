#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path
import cv2

TEMPLATE = """[Sequence]
name={name}
imDir=img1
frameRate={fps}
seqLength={nframes}
imWidth={w}
imHeight={h}
imExt=.jpg
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., data/meta/mot/val10")
    ap.add_argument("--fps", type=float, required=True, help="e.g., 10")
    args = ap.parse_args()

    root = Path(args.root)
    for seqdir in sorted([d for d in root.iterdir() if d.is_dir()]):
        imgdir = seqdir / "img1"
        imgs = sorted(glob.glob(str(imgdir / "*.jpg")))
        if not imgs:
            print(f"[skip] no frames in {imgdir}")
            continue
        # peek first frame for size
        img = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed to read first frame in {imgdir}")
            continue
        h, w = img.shape[:2]
        nframes = len(imgs)
        ini = TEMPLATE.format(name=seqdir.name, fps=args.fps, nframes=nframes, w=w, h=h)
        (seqdir / "seqinfo.ini").write_text(ini)
        print(f"[ok] {seqdir.name}: {nframes} frames, {w}x{h}, fps={args.fps}")
if __name__ == "__main__":
    main()


# python3 tools/build_mot_seqinfo.py --root data/meta/mot/val10 --fps 10