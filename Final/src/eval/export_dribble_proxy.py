#!/usr/bin/env python3
"""
Export 'dribble_proxy' segments from rules_actions outputs to CSV and
produce a per-clip visualization of ball-to-owner distance over time.

Inputs
- Actions JSON per sequence:
    results/actions_auto/<split>/<seq>.json
  Expected (if present):
    {
      "sequence": "...",
      "events": [...],
      "dribble_proxy": [
         {"player_id": 7, "t_start": 5, "t_end": 10,
          "band_share": 0.63, "var": 310.5}
      ]
    }

- MOT tracking results to reconstruct distance series:
    results/track/<tracker>/<split>/<seq>/<seq>.txt
  Format: frame,id,x,y,w,h,conf,cls,vis   (cls: player=--player_cls, ball=--ball_cls)

Outputs
- CSV per sequence:
    results/analysis/dribble_proxy/<split>/<seq>_dribproxy.csv
  Columns: sequence,player_id,t_start,t_end,frames,band_share,var

- Combined CSV across processed sequences:
    results/analysis/dribble_proxy/<split>/_all_dribproxy.csv

- Viz PNG per sequence:
    results/analysis/dribble_proxy/<split>/<seq>_dribproxy.png
  Shows distance (px) between ball and each proxied owner's center over frames,
  with near/far bands shaded and proxy segments highlighted.

Usage (examples)
  # Process all v1_* clips in val split (botsort tracks)
  python3 src/eval/export_dribble_proxy.py \
    --split val --tracker botsort --game_id v1 \
    --near_px 40 --far_px 100

  # Process a single sequence
  python3 src/eval/export_dribble_proxy.py \
    --split val --tracker botsort --seq v1_p4 \
    --near_px 40 --far_px 100
"""
import argparse, csv, glob, json, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- IO helpers (aligned with your codebase) -----------------
def load_mot(mot_file: Path):
    """Return dict frame -> list of (tid, cls, x,y,w,h, conf)."""
    by_f = defaultdict(list)
    if not mot_file.exists():
        return by_f
    for line in mot_file.read_text().splitlines():
        if not line.strip():
            continue
        toks = re.split(r"[,\s]+", line.strip())
        if len(toks) < 8:
            continue
        fr   = int(float(toks[0])); tid = int(float(toks[1]))
        x,y,w,h = map(float, toks[2:6])
        conf = float(toks[6])
        cls  = int(float(toks[7]))
        by_f[fr].append((tid, cls, x,y,w,h, conf))
    return by_f

def cxcy(x,y,w,h):
    return (x + w/2.0, y + h/2.0)

def dist2d(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

# ----------------- main logic -----------------
def export_for_seq(seq:str,
                   actions_json:Path,
                   mot_file:Path,
                   out_dir:Path,
                   player_cls:int,
                   ball_cls:int,
                   near_px:float,
                   far_px:float,
                   plot_top:int):
    """
    Returns number of segments exported.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not actions_json.exists():
        print(f"[WARN] no actions json: {actions_json}")
        return 0

    obj = json.loads(actions_json.read_text())
    segs = obj.get("dribble_proxy", [])
    if not segs:
        # still write an empty CSV with header for traceability
        out_csv = out_dir / f"{seq}_dribproxy.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence","player_id","t_start","t_end","frames","band_share","var"])
        # also create a small placeholder viz
        fig = plt.figure(figsize=(10, 3))
        plt.title(f"{seq}: dribble_proxy (none)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{seq}_dribproxy.png", dpi=180)
        plt.close(fig)
        print(f"[{seq}] no dribble_proxy; wrote empty CSV and placeholder viz")
        return 0

    # Write per-seq CSV
    out_csv = out_dir / f"{seq}_dribproxy.csv"
    rows = []
    for s in segs:
        rows.append([
            seq,
            int(s.get("player_id",-1)),
            int(s.get("t_start",0)),
            int(s.get("t_end",0)),
            int(max(0, s.get("t_end",0) - s.get("t_start",0) + 1)),
            float(s.get("band_share",0.0)),
            float(s.get("var",0.0)),
        ])
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence","player_id","t_start","t_end","frames","band_share","var"])
        w.writerows(rows)
    print(f"[{seq}] wrote CSV: {out_csv}")

    # Build distance series from MOT (ball vs player centers)
    tracks = load_mot(mot_file)
    if not tracks:
        # viz without series
        fig = plt.figure(figsize=(12, 3))
        plt.title(f"{seq}: no MOT tracks found, cannot plot distances")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{seq}_dribproxy.png", dpi=180)
        plt.close(fig)
        print(f"[{seq}] no MOT; wrote placeholder viz")
        return len(segs)

    frames = sorted(tracks.keys())
    # Build per-frame centers
    ball_c = {}
    player_c = defaultdict(dict)  # frame -> {tid: (cx,cy)}
    for f in frames:
        # pick highest-conf ball for this frame
        balls = [b for b in tracks[f] if b[1]==ball_cls]
        if balls:
            b = max(balls, key=lambda z: z[-1])
            ball_c[f] = cxcy(b[2],b[3],b[4],b[5])
        for p in [pp for pp in tracks[f] if pp[1]==player_cls]:
            player_c[f][p[0]] = cxcy(p[2],p[3],p[4],p[5])

    # Score segments to decide which to plot if too many
    # score = band_share * log(1+var) * length
    scored = []
    for s in segs:
        length = int(max(0, s.get("t_end",0) - s.get("t_start",0) + 1))
        score = float(s.get("band_share",0.0)) * np.log1p(float(s.get("var",0.0))) * max(1,length)
        scored.append((score, s))
    scored.sort(key=lambda z: -z[0])

    # Plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f"{seq}: ball-owner distance (px) with near/far bands")
    ax.set_xlabel("frame")
    ax.set_ylabel("distance (px)")
    # shaded band
    ymin, ymax = 0, max(200.0, far_px*1.5)
    ax.set_ylim(ymin, ymax)
    ax.axhspan(0, near_px, facecolor="#e8ffe8", alpha=0.45, label=f"near ≤ {near_px:.0f}")
    ax.axhspan(near_px, far_px, facecolor="#fff8d9", alpha=0.40, label=f"neutral {near_px:.0f}–{far_px:.0f}")
    ax.axhspan(far_px, ymax, facecolor="#ffeaea", alpha=0.35, label=f"far ≥ {far_px:.0f}")

    cmap = plt.get_cmap("tab10")
    plotted = 0
    for _, s in scored:
        if plotted >= plot_top:
            break
        pid = int(s.get("player_id",-1))
        t0  = int(s.get("t_start",0))
        t1  = int(s.get("t_end",0))
        xs, ys = [], []
        for f in range(t0, t1+1):
            bc = ball_c.get(f)
            pc = player_c.get(f, {}).get(pid)
            if bc and pc:
                xs.append(f)
                ys.append(dist2d(bc, pc))
        if len(xs) >= 2:
            ax.plot(xs, ys, marker="o", ms=3,
                    lw=2, alpha=0.9, color=cmap(plotted),
                    label=f"pid {pid} [{t0}-{t1}] share={s.get('band_share',0):.2f} var={s.get('var',0):.0f}")
            # highlight segment span
            ax.axvspan(t0, t1, color=cmap(plotted), alpha=0.15)
            plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No distance samples for proxied segments",
                transform=ax.transAxes, ha="center", va="center")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"{seq}_dribproxy.png", dpi=180)
    plt.close(fig)
    print(f"[{seq}] wrote viz: {out_dir / (seq + '_dribproxy.png')}")
    return len(segs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val"], default="val")
    ap.add_argument("--tracker", choices=["bytetrack","botsort"], default="botsort")
    ap.add_argument("--seq", default=None, help="single sequence (e.g., v1_p4)")
    ap.add_argument("--game_id", default=None, help="prefix to filter sequences (e.g., v1)")
    ap.add_argument("--player_cls", type=int, default=0)
    ap.add_argument("--ball_cls",   type=int, default=1)
    ap.add_argument("--actions_root", default="results/actions_auto")
    ap.add_argument("--res_root",     default="results/track")
    ap.add_argument("--out_root",     default="results/analysis/dribble_proxy")
    ap.add_argument("--near_px", type=float, default=40.0)
    ap.add_argument("--far_px",  type=float, default=100.0)
    ap.add_argument("--plot_top", type=int, default=3, help="max segments to plot per clip")
    args = ap.parse_args()

    actions_split = Path(args.actions_root)/args.split
    res_split = Path(args.res_root)/args.tracker/args.split
    out_split = Path(args.out_root)/args.split
    out_split.mkdir(parents=True, exist_ok=True)

    # discover sequences from actions JSONs
    jsons = sorted(actions_split.glob("*.json"))
    seq_list = []
    for jf in jsons:
        seq = jf.stem
        if args.seq and seq != args.seq:
            continue
        if args.game_id and not seq.startswith(f"{args.game_id}_"):
            continue
        seq_list.append((seq, jf))

    if not seq_list:
        print(f"[WARN] no sequences found under {actions_split} (filters: seq={args.seq} game_id={args.game_id})")
        return

    # process & collect combined CSV
    combined_rows = []
    for seq, jf in seq_list:
        mot_file = res_split / seq / f"{seq}.txt"
        nseg = export_for_seq(seq, jf, mot_file, out_split,
                              args.player_cls, args.ball_cls,
                              args.near_px, args.far_px, args.plot_top)
        # append rows from per-seq CSV into combined
        per_csv = out_split / f"{seq}_dribproxy.csv"
        if per_csv.exists():
            with per_csv.open() as f:
                r = csv.DictReader(f)
                for row in r:
                    combined_rows.append([
                        row.get("sequence", seq),
                        row.get("player_id", ""),
                        row.get("t_start",""),
                        row.get("t_end",""),
                        row.get("frames",""),
                        row.get("band_share",""),
                        row.get("var",""),
                    ])

    # write combined
    if combined_rows:
        all_csv = out_split / "_all_dribproxy.csv"
        with all_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence","player_id","t_start","t_end","frames","band_share","var"])
            w.writerows(combined_rows)
        print(f"[OK] wrote combined CSV: {all_csv}")

if __name__ == "__main__":
    main()