#!/usr/bin/env python3
"""
Compute ball ownership per frame from MOT results.

Inputs:
  - MOT images: data/meta/mot/<split>/<seq>/img1
  - Tracker results: results/track/<tracker>/<split>/<seq>/<seq>.txt
Optional:
  - Court coords: data/meta/tracks_xy/<seq>.csv  (if available, used for distances)
Output:
  - data/meta/ownership/<seq>.csv   (frame, owner_tid or -1)
"""
import argparse, csv
from pathlib import Path
import numpy as np, math

def read_mot(path):
    M = {}
    for row in csv.reader(open(path)):
        if len(row) < 8: 
            continue
        fr = int(float(row[0])); tid=int(float(row[1]))
        x,y,w,h = map(float, row[2:6]); conf=float(row[6]); cls=int(float(row[7]))
        M.setdefault(fr, []).append((tid,x,y,w,h,conf,cls))
    return M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--res_root", default="results/track")
    ap.add_argument("--mot_root", default="data/meta/mot")
    ap.add_argument("--out_root", default="data/meta/ownership")
    ap.add_argument("--pix_gate_frac", type=float, default=0.035, help="ball->player max distance as frac of image diagonal")
    args = ap.parse_args()

    seq = args.seq
    res = Path(args.res_root)/args.tracker/args.split/seq/f"{seq}.txt"
    mot = read_mot(res)

    # try to read image size from seqinfo.ini
    ini = Path(args.mot_root)/args.split/seq/"seqinfo.ini"
    W = H = None
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith("imWidth="):  W = int(line.split("=")[1])
            if line.startswith("imHeight="): H = int(line.split("=")[1])
    diag = math.hypot(W or 1920, H or 1080)
    gate = args.pix_gate_frac * diag

    out_dir = Path(args.out_root); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir/f"{seq}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame","owner_tid"])
        frames = sorted(mot.keys())
        for fr in frames:
            balls   = [(tid,x,y,w,h,conf,cls) for (tid,x,y,w,h,conf,cls) in mot[fr] if cls==2]
            players = [(tid,x,y,w,h,conf,cls) for (tid,x,y,w,h,conf,cls) in mot[fr] if cls==1]
            if not balls or not players:
                w.writerow([fr, -1]); continue
            # pick the most confident ball if multiple
            tid_b, xb,yb,wb,hb,cb,_ = sorted(balls, key=lambda z:-z[5])[0]
            cx_b, cy_b = xb+wb/2, yb+hb/2
            best = (-1, 1e9)
            for (tid,x,y,w_,h_,c_,_) in players:
                cx, cy = x+w_/2, y+h_/2
                d = math.hypot(cx-cx_b, cy-cy_b)
                if d < best[1]:
                    best = (tid, d)
            w.writerow([fr, best[0] if best[1] <= gate else -1])
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()