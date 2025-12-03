#!/usr/bin/env python3
import argparse, re, csv
from pathlib import Path

def read_seqinfo(ini: Path):
    W, H = 1280, 720
    if ini.exists():
        for line in ini.read_text().splitlines():
            if line.startswith('imWidth='):  W = int(line.split('=')[1])
            if line.startswith('imHeight='): H = int(line.split('=')[1])
    return W, H

def yolo_label_to_mot_row(frame_i: int, line: str, W: int, H: int):
    """
    Accepts UL track label lines in either format:
      6 fields: cls cx cy w h id
      7 fields: cls cx cy w h conf id
    All coords are normalized; we output MOT in *pixels*:
      frame,id,x,y,w,h,conf,cls,vis
    """
    parts = [p for p in line.strip().split() if p]
    if len(parts) not in (6, 7):
        return None

    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])

    if len(parts) == 6:
        conf = 1.0
        tid  = int(float(parts[5]))
    else:  # 7 fields
        conf = float(parts[5])
        tid  = int(float(parts[6]))

    x = (cx - w / 2.0) * W
    y = (cy - h / 2.0) * H
    w *= W
    h *= H
    return [frame_i, tid, x, y, w, h, conf, cls, 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ultra_root", default="results/track")
    ap.add_argument("--tracker", required=True, choices=["botsort", "bytetrack"])
    ap.add_argument("--split", required=True, choices=["val", "train", "val10"])
    ap.add_argument("--mot_out", required=True)     # e.g. results/track/botsort/val
    ap.add_argument("--mot_gt",  default="data/meta/mot")  # has <split>/<seq>/seqinfo.ini
    args = ap.parse_args()

    ultra_root = Path(args.ultra_root)
    mot_out = Path(args.mot_out); mot_out.mkdir(parents=True, exist_ok=True)

    # Match run folders like: botsort_val_v1_p4 / bytetrack_val_v3_p6
    pat = re.compile(rf"^{args.tracker}_2ps_(?P<seq>.+)$")
    run_dirs = [d for d in ultra_root.iterdir() if d.is_dir() and pat.match(d.name)]
    if not run_dirs:
        raise SystemExit(f"No UL runs found like {args.tracker}_{args.split}_* under {ultra_root}")

    for rd in sorted(run_dirs, key=lambda p: p.name):
        seq = pat.match(rd.name).group("seq")
        labels_dir = rd / "labels"
        if not labels_dir.exists():
            print(f"[WARN] no labels/ in {rd}; skip"); continue

        ini = Path(args.mot_gt) / args.split / seq / "seqinfo.ini"
        W, H = read_seqinfo(ini)

        out_dir = mot_out / seq
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{seq}.txt"

        rows = []
        for lab in sorted(labels_dir.glob("*.txt")):
            frame_i = int(lab.stem)  # '000123' -> 123
            for line in lab.read_text().splitlines():
                r = yolo_label_to_mot_row(frame_i, line, W, H)
                if r is not None:
                    rows.append(r)

        with out_file.open("w", newline="") as f:
            w = csv.writer(f)
            for r in rows:
                w.writerow(r)
        print(f"[OK] {seq}: wrote {out_file} ({len(rows)} rows)")

if __name__ == "__main__":
    main()