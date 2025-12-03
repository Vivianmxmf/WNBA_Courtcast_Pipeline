#!/usr/bin/env python3
"""
Dump per-frame owner↔ball distance (px and ft), owner continuity, and scale context.

Inputs
- MOT tracks:
    results/track/<tracker>/<split>/<seq>/<seq>.txt
  Format: frame,id,x,y,w,h,conf,cls,vis   (player_cls=0, ball_cls=1 by default)

- Optional court projections (if present):
    results/court_tracks/<split>/<seq>.csv
  Format: frame,tid,cls,x,y,conf,half   (x,y are in feet)

Outputs (per sequence)
- results/analysis/owner_dist/<split>/<seq>_owner_dist.csv
  Columns:
    frame,owner_id,dist_px,ball_px,ball_py,own_px,own_py,
    ball_speed_px,owner_bbox_h,
    dist_ft,ball_x_ft,ball_y_ft,own_x_ft,own_y_ft,ball_speed_ft

Notes
- Owner assignment uses hysteresis: if a player is within own_px it becomes owner; it stays owner
  until all candidates exceed (own_px + own_hys). You should pass the same thresholds you use
  in rules_actions.py so diagnostics are consistent.
"""
import argparse, csv, re
from pathlib import Path
from collections import defaultdict, deque
import numpy as np

def load_mot(mot_file, player_cls, ball_cls):
    by_f = defaultdict(list)
    for line in mot_file.read_text().splitlines():
        if not line.strip(): continue
        t = re.split(r"[,\s]+", line.strip())
        if len(t) < 8: continue
        f   = int(float(t[0])); tid = int(float(t[1]))
        x,y,w,h = map(float, t[2:6])
        conf = float(t[6]); cls = int(float(t[7]))
        by_f[f].append((tid,cls,x,y,w,h,conf))
    return by_f

def centers(tracks, player_cls, ball_cls):
    frames = sorted(tracks.keys())
    ball_c = {}
    player_c = defaultdict(dict)
    bbox_h  = defaultdict(dict)
    for f in frames:
        balls = [b for b in tracks[f] if b[1]==ball_cls]
        if balls:
            b = max(balls, key=lambda z: z[-1])
            ball_c[f] = (b[2]+b[4]/2.0, b[3]+b[5]/2.0)
        for p in [pp for pp in tracks[f] if pp[1]==player_cls]:
            cx, cy = p[2]+p[4]/2.0, p[3]+p[5]/2.0
            player_c[f][p[0]] = (cx, cy)
            bbox_h[f][p[0]]   = p[5]
    return frames, ball_c, player_c, bbox_h

def smooth_series(xs, k):
    if k<=1 or not xs: return xs[:]
    q = deque(maxlen=k)
    out = []
    for v in xs:
        q.append(v); out.append(sum(q)/len(q))
    return out

def load_court_csv(csv_path):
    if not csv_path.exists(): return {}
    idx = {}
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                fno = int(row["frame"]); tid = int(row["tid"])
                xf  = float(row["x"]);   yf  = float(row["y"])
            except: continue
            idx[(fno, tid, int(row["cls"]))] = (xf, yf)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val","val10"], default="val")
    ap.add_argument("--tracker", choices=["bytetrack","botsort"], default="botsort")
    ap.add_argument("--game_id", default=None)   # e.g., v1
    ap.add_argument("--seq", default=None)       # e.g., v1_p4
    ap.add_argument("--player_cls", type=int, default=0)
    ap.add_argument("--ball_cls",   type=int, default=1)
    ap.add_argument("--own_px",  type=float, default=140.0)
    ap.add_argument("--own_hys", type=float, default=20.0)
    ap.add_argument("--ball_smooth", type=int, default=2)
    ap.add_argument("--mot_root", default="results/track")
    ap.add_argument("--ct_root",  default="results/court_tracks")
    ap.add_argument("--out_root", default="results/analysis/owner_dist")
    args = ap.parse_args()

    mot_split = Path(args.mot_root)/args.tracker/args.split
    ct_split  = Path(args.ct_root)/args.split
    out_split = Path(args.out_root)/args.split
    out_split.mkdir(parents=True, exist_ok=True)

    # discover sequences
    seqs = []
    for d in sorted(mot_split.glob("*")):
        if not d.is_dir(): continue
        seq = d.name
        if args.seq and seq!=args.seq: continue
        if args.game_id and not seq.startswith(f"{args.game_id}_"): continue
        seqs.append(seq)

    for seq in seqs:
        mot_file = mot_split/seq/f"{seq}.txt"
        if not mot_file.exists():
            print(f"[WARN] no MOT: {mot_file}"); continue

        tracks = load_mot(mot_file, args.player_cls, args.ball_cls)
        if not tracks:
            print(f"[WARN] empty MOT: {seq}"); continue

        frames, ball_c, player_c, bbox_h = centers(tracks, args.player_cls, args.ball_cls)

        # simple owner hysteresis
        owner = {}
        cur_owner = None
        for f in frames:
            bc = ball_c.get(f)
            if not bc:
                owner[f] = -1; continue
            # distances to players this frame
            dists = []
            for tid, pc in player_c[f].items():
                d = np.hypot(pc[0]-bc[0], pc[1]-bc[1])
                dists.append((d, tid))
            dists.sort()
            new_owner = None
            if dists and dists[0][0] <= args.own_px:
                new_owner = dists[0][1]
            # hysteresis: keep current unless it's clearly broken
            if cur_owner is None:
                cur_owner = new_owner
            else:
                if cur_owner in player_c[f]:
                    d_cur = np.hypot(player_c[f][cur_owner][0]-bc[0],
                                     player_c[f][cur_owner][1]-bc[1])
                else:
                    d_cur = 1e9
                # if current far beyond (own_px + own_hys), allow switch
                if d_cur > (args.own_px + args.own_hys) and new_owner is not None:
                    cur_owner = new_owner
                # else keep cur_owner
            owner[f] = cur_owner if cur_owner is not None else -1

        # ball speed (px/frame), smoothed
        ball_xy = [ball_c.get(f) for f in frames]
        vx = [0.0]
        for i in range(1, len(frames)):
            if ball_xy[i] and ball_xy[i-1]:
                vx.append(float(np.hypot(ball_xy[i][0]-ball_xy[i-1][0],
                                         ball_xy[i][1]-ball_xy[i-1][1])))
            else:
                vx.append(0.0)
        vx = smooth_series(vx, args.ball_smooth)

        # optional court-space join
        ct_idx = load_court_csv(ct_split/f"{seq}.csv")

        out_csv = out_split/f"{seq}_owner_dist.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame","owner_id",
                        "dist_px","ball_px","ball_py","own_px","own_py",
                        "ball_speed_px","owner_bbox_h",
                        "dist_ft","ball_x_ft","ball_y_ft","own_x_ft","own_y_ft","ball_speed_ft"])
            for i, fno in enumerate(frames):
                bc = ball_c.get(fno)
                pid = owner.get(fno, -1)
                pc  = player_c.get(fno, {}).get(pid)
                dpx = float(np.hypot((pc[0]-bc[0]), (pc[1]-bc[1]))) if (bc and pc and pid!=-1) else ""
                bh  = bbox_h.get(fno, {}).get(pid, "")
                # court coords (if available)
                b_ft = ct_idx.get((fno, 0, args.ball_cls))   # (frame,tid,cls) — ball tid isn't 0, so this will likely miss
                # Instead, search any ball tid on that frame:
                bft = None
                for (ff, tid, cls), xy in ct_idx.items():
                    if ff==fno and cls==args.ball_cls:
                        bft = xy; break
                oft = ct_idx.get((fno, pid, args.player_cls)) if pid!=-1 else None
                dft = ""
                bsft = ""
                if bft and oft:
                    dft = float(np.hypot(bft[0]-oft[0], bft[1]-oft[1]))
                if i>0 and bft and (prev:= [xy for (ff,_,cls),xy in ct_idx.items() if ff==frames[i-1] and cls==args.ball_cls]):
                    bsft = float(np.hypot(bft[0]-prev[0][0], bft[1]-prev[0][1]))
                w.writerow([
                    fno, pid,
                    dpx,
                    bc[0] if bc else "", bc[1] if bc else "",
                    pc[0] if pc else "", pc[1] if pc else "",
                    vx[i] if i<len(vx) else "",
                    bh,
                    dft if dft!="" else "",
                    bft[0] if bft else "", bft[1] if bft else "",
                    oft[0] if oft else "", oft[1] if oft else "",
                    bsft if bsft!="" else ""
                ])
        print(f"[OK] wrote {out_csv}")

if __name__ == "__main__":
    main()