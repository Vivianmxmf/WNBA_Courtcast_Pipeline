#!/usr/bin/env python3
"""
Sample player crops from tracker outputs to build jersey datasets.

Inputs:
  - MOT images: data/meta/mot/{train|val}/<seq>/img1/000001.png ...
  - Tracker results: results/track/<tracker>/<split>/<seq>/<seq>.txt
Output:
  - data/meta/team_crops/<game_or_seq>/<track_id>/*.png

Usage:
  python src/team/make_crops.py --tracker botsort --split val --stride 10 \
    --mode upper --scale_w 0.6 --upper_top 0.10 --upper_bot 0.60 --class_id 0
"""
import argparse, csv, sys
from pathlib import Path
import cv2, os

def read_mot_results(path):
    """Return dict: frame -> list of (tid, x, y, w, h, conf, cls).
       CSV cols: frame,id,x,y,w,h,conf,class[,vis]"""
    M = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 8:
                continue
            fr  = int(float(row[0]))
            tid = int(float(row[1]))
            x, y, w, h = map(float, row[2:6])
            conf = float(row[6])
            cls  = int(float(row[7]))
            M.setdefault(fr, []).append((tid, x, y, w, h, conf, cls))
    return M

def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
    return int(x0), int(y0), int(x1), int(y1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, help="bytetrack or botsort")
    ap.add_argument("--split",   default="val", choices=["train","val", "val10"])
    ap.add_argument("--mot_root", default="data/meta/mot")
    ap.add_argument("--res_root", default="results/track")
    ap.add_argument("--out_root", default="data/meta/team_crops")
    ap.add_argument("--stride", type=int, default=10, help="sample every N frames")
    ap.add_argument("--player_cls", type=int, default=0, help="class id for players in MOT (default 0)")
    ap.add_argument("--seq", default=None, help="optional single sequence to process")
    ap.add_argument("--game_id", default=None, help="optional folder name override")

    # --- crop controls ---
    ap.add_argument("--mode", choices=["upper","center","tight"], default="upper",
                    help="upper=torso band, center=centered shrink, tight=full box with pad")
    ap.add_argument("--scale_w", type=float, default=0.6,
                    help="relative width for upper/center (0<scale<=1)")
    ap.add_argument("--scale_h", type=float, default=0.5,
                    help="relative height for center (ignored in upper)")
    ap.add_argument("--upper_top", type=float, default=0.10,
                    help="top fraction of bbox for upper mode (0..1)")
    ap.add_argument("--upper_bot", type=float, default=0.60,
                    help="bottom fraction of bbox for upper mode (0..1)")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="only used in tight mode: pad as fraction of w/h")

    # --- filtering ---
    ap.add_argument("--class_id", type=int, default=0,
                    help="player class id in MOT files (yours is 0)")
    ap.add_argument("--conf_min", type=float, default=0.0,
                    help="drop boxes below this conf")

    args = ap.parse_args()

    mot_split = Path(args.mot_root)/args.split
    res_split = Path(args.res_root)/args.tracker/args.split
    out_root  = Path(args.out_root)

    if not mot_split.exists():
        sys.exit(f"[ERR] MOT split not found: {mot_split}")
    if not res_split.exists():
        sys.exit(f"[ERR] Tracker split not found: {res_split}")

    seqs = [mot_split / args.seq] if args.seq else [d for d in mot_split.iterdir() if d.is_dir()]
    if not seqs:
        sys.exit(f"[ERR] No sequences under {mot_split}")

    total_crops = 0
    for seq_dir in sorted(seqs):
        seq = seq_dir.name
        game_folder = args.game_id or seq
        img_dir = seq_dir/"img1"

        res_file = res_split/seq/f"{seq}.txt"
        if not res_file.exists():
            alt = res_split/f"{seq}.txt"
            if alt.exists():
                res_file = alt
        if not res_file.exists():
            print(f"[WARN] no result: {res_file}")
            continue

        mot = read_mot_results(res_file)
        if not mot:
            print(f"[WARN] empty MOT result: {res_file}")
            continue

        # first = next(iter(img_dir.glob("000001.*")), None)
        # if first is None:
        #     print(f"[WARN] no frames in {img_dir}")
        #     continue
        # ext = first.suffix
        imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
        if not imgs:
            print(f"[WARN] no frames in {img_dir}")
            continue
        ext = imgs[0].suffix


        frames = sorted(mot.keys())
        crops_this_seq = 0
        for fr in frames[::max(1, args.stride)]:
            im_path = img_dir/f"{fr:06d}{ext}"
            img = cv2.imread(str(im_path))
            if img is None:
                for e in (".png",".jpg",".jpeg"):
                    p = img_dir/f"{fr:06d}{e}"
                    if p.exists():
                        img = cv2.imread(str(p)); im_path = p; ext = e
                        break
            if img is None: 
                continue

            H, W = img.shape[:2]
            for (tid, x, y, w, h, conf, cls) in mot.get(fr, []):
                if cls != args.player_cls or conf < args.conf_min:
                    continue

                # --- compute shrunken crop box ---
                if args.mode == "upper":
                    # torso band: keep width shrink around center; vertical slice [upper_top, upper_bot]
                    cx = x + 0.5 * w
                    new_w = max(2.0, w * args.scale_w)
                    x0 = cx - 0.5 * new_w
                    x1 = cx + 0.5 * new_w
                    y0 = y + args.upper_top * h
                    y1 = y + args.upper_bot * h

                elif args.mode == "center":
                    cx, cy = x + 0.5 * w, y + 0.5 * h
                    new_w = max(2.0, w * args.scale_w)
                    new_h = max(2.0, h * args.scale_h)
                    x0 = cx - 0.5 * new_w
                    x1 = cx + 0.5 * new_w
                    y0 = cy - 0.5 * new_h
                    y1 = cy + 0.5 * new_h

                else:  # tight
                    x0 = x - args.pad * w
                    y0 = y - args.pad * h
                    x1 = x + w + args.pad * w
                    y1 = y + h + args.pad * h

                x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
                if x1 <= x0 or y1 <= y0:
                    continue
                if (x1 - x0) < 12 or (y1 - y0) < 12:  # avoid tiny/noisy crops
                    continue

                crop = img[y0:y1, x0:x1]
                out_dir = out_root/game_folder/f"{seq}_{tid}"
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir/f"{seq}_{fr:06d}.png"), crop)
                crops_this_seq += 1
                total_crops += 1

        print(f"[OK] {seq}: wrote {crops_this_seq} crops -> {out_root/(args.game_id or seq)}")
    print(f"[DONE] total crops written: {total_crops}")

if __name__ == "__main__":
    main()