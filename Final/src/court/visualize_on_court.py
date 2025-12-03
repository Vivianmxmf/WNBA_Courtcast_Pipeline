#!/usr/bin/env python3
"""
Overlay projected tracks/events on a top-down court PNG.

Supports both CSV schemas:
 A) frame,tid,cls,x,y,conf,half
 B) frame,id,cls,x_ft,y_ft,in_bounds
"""

import argparse, json, glob, csv
from pathlib import Path
import numpy as np, cv2

# Court in feet
L, W = 94.0, 50.0
X_MIN, X_MAX = -L/2, L/2
Y_MIN, Y_MAX = -W/2, W/2

# ---------- coord helpers ----------
def ft_to_px(xf, yf, Wpx, Hpx, pad):
    s = min((Wpx - 2*pad)/L, (Hpx - 2*pad)/W)
    u = Wpx/2 + xf*s
    v = Hpx/2 - yf*s
    return int(round(u)), int(round(v))

def _row_xy_ft(row):
    if "x_ft" in row and "y_ft" in row:
        return float(row["x_ft"]), float(row["y_ft"])
    return float(row["x"]), float(row["y"])

def _row_id(row):
    return int(float(row["id"])) if "id" in row else int(float(row["tid"]))

def _row_inb(row):
    v = row.get("in_bounds", None)
    return int(v) if v not in (None, "") else 1
# ------------------------------------

def load_csv_points(csv_glob, want_cls=None, clip_out=True):
    """Return list of (seq, frame, tid, cls, xf, yf)."""
    pts, clipped = [], 0
    for fp in sorted(glob.glob(csv_glob)):
        seq = Path(fp).stem
        with open(fp) as f:
            r = csv.DictReader(f)
            for row in r:
                cls = int(float(row["cls"]))
                if (want_cls is not None) and (cls != want_cls):
                    continue
                xf, yf = _row_xy_ft(row)
                if clip_out and not (X_MIN <= xf <= X_MAX and Y_MIN <= yf <= Y_MAX):
                    clipped += 1
                    continue
                tid = _row_id(row)
                pts.append((seq, int(float(row["frame"])), tid, cls, xf, yf))
    return pts, clipped

def draw_scatter(
    bg, pad, pts, which="both", teams=None,
    alpha=1.0, r_player=7, r_ball=10, edge_th=2
):
    """
    Strong, high-contrast colors + black outline.
    teams: optional dict tid->'home'/'away' for player coloring.
    """
    out = bg.copy()
    H, Wpx = out.shape[:2]

    # vivid dark palette (BGR)
    C_HOME = (255, 80, 40)   # deep blue
    C_AWAY = (40, 140, 255)  # orange
    C_PLAYER = (180, 60, 60) # fallback magenta-ish
    C_BALL = (0, 220, 255)   # yellow
    EDGE = (0, 0, 0)         # black outline

    for (_,_,tid,cls,xf,yf) in pts:
        if which == "player" and cls != 0: continue
        if which == "ball"   and cls != 1: continue
        u, v = ft_to_px(xf, yf, Wpx, H, pad)
        if cls == 1:
            color, rad = C_BALL, r_ball
        else:
            if teams:
                lab = teams.get(str(tid)) or teams.get(tid)
                color = C_HOME if lab == "home" else (C_AWAY if lab == "away" else C_PLAYER)
            else:
                color = C_PLAYER
            rad = r_player
        # edge + fill for contrast on bright background
        cv2.circle(out, (u, v), rad + edge_th, EDGE, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (u, v), rad, color, -1, lineType=cv2.LINE_AA)

    # Standard blending
    if alpha >= 1.0: return out
    return cv2.addWeighted(out, alpha, bg, 1.0-alpha, 0)

def draw_heatmap(bg, pad, pts, sigma_px=24, which_cls=0, colormap=cv2.COLORMAP_TURBO, alpha=0.65):
    H, Wpx = bg.shape[:2]
    acc = np.zeros((H, Wpx), np.float32)
    for (_,_,_,cls,xf,yf) in pts:
        if cls != which_cls: continue
        u, v = ft_to_px(xf, yf, Wpx, H, pad)
        if 0 <= u < Wpx and 0 <= v < H:
            acc[v, u] += 1.0
    k = int(max(3, sigma_px*3)//2*2 + 1)
    blur = cv2.GaussianBlur(acc, (k, k), sigma_px, sigma_px, borderType=cv2.BORDER_REPLICATE)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(norm, colormap)
    return cv2.addWeighted(bg, 1.0, color, alpha, 0)

def index_points_by_seq_frame_id(csv_glob):
    idx = {}
    for fp in sorted(glob.glob(csv_glob)):
        seq = Path(fp).stem
        with open(fp) as f:
            for row in csv.DictReader(f):
                xf, yf = _row_xy_ft(row)
                tid    = _row_id(row)
                idx[(seq, int(float(row["frame"])), tid)] = (xf, yf)
    return idx

def draw_passes(bg, pad, csv_glob, events_glob, color=(0, 255, 0), alpha=1.0):
    """Draw arrows for passes (Green)."""
    H, Wpx = bg.shape[:2]
    out = bg.copy()
    idx = index_points_by_seq_frame_id(csv_glob)

    core_thick   = 7          
    outline_thick= core_thick + 4
    tip_frac     = 0.18       
    endpoint_px  = core_thick + 2
    shorten_px   = 10         

    def shrink(u0, v0, u1, v1, m=shorten_px):
        du, dv = float(u1 - u0), float(v1 - v0)
        norm = (du*du + dv*dv) ** 0.5
        if norm < 1e-3: return u0, v0, u1, v1
        offu, offv = int(round(m * du / norm)), int(round(m * dv / norm))
        return u0 + offu, v0 + offv, u1 - offu, v1 - offv

    n_draw = 0
    for jf in sorted(glob.glob(events_glob)):
        stem = Path(jf).stem
        seq = stem.replace("_pass_pairs", "") 

        blob = json.loads(Path(jf).read_text())
        if isinstance(blob, list):
            E = [x["a"] for x in blob if isinstance(x, dict) and "a" in x]
            if not E and blob: E = blob
        else:
            E = blob.get("events", [])

        for e in E:
            if e.get("type") != "pass": continue
            fr0, fr1 = e.get("t_release"), e.get("t_catch")
            id0, id1 = e.get("from_id"), e.get("to_id")
            
            p0, p1 = idx.get((seq, fr0, id0)), idx.get((seq, fr1, id1))
            if (not p0 or not p1) and ("from_xy" in e and "to_xy" in e):
                p0, p1 = (e["from_xy"][0], e["from_xy"][1]), (e["to_xy"][0], e["to_xy"][1])
            if not (p0 and p1): continue

            u0, v0 = ft_to_px(p0[0], p0[1], Wpx, H, pad)
            u1, v1 = ft_to_px(p1[0], p1[1], Wpx, H, pad)
            u0, v0, u1, v1 = shrink(u0, v0, u1, v1)

            cv2.arrowedLine(out, (u0, v0), (u1, v1), (0, 0, 0), outline_thick, tipLength=tip_frac, line_type=cv2.LINE_AA)
            cv2.arrowedLine(out, (u0, v0), (u1, v1), color,           core_thick,    tipLength=tip_frac, line_type=cv2.LINE_AA)

            cv2.circle(out, (u0, v0), endpoint_px+2, (0, 0, 0), -1, lineType=cv2.LINE_AA)              
            cv2.circle(out, (u0, v0), endpoint_px,   (255, 255, 255), -1, lineType=cv2.LINE_AA)        
            cv2.circle(out, (u1, v1), endpoint_px+2, (0, 0, 0), -1, lineType=cv2.LINE_AA)              
            cv2.circle(out, (u1, v1), endpoint_px,   color,            -1, lineType=cv2.LINE_AA)       
            n_draw += 1

    print(f"[passes] drew {n_draw} arrows")
    if alpha >= 1.0: return out
    return cv2.addWeighted(out, alpha, bg, 1.0-alpha, 0)

def draw_shots(bg, pad, csv_glob, events_glob, color=(0, 0, 255), alpha=1.0):
    """Draw crosses for shots (Red)."""
    H, Wpx = bg.shape[:2]
    out = bg.copy()
    idx = index_points_by_seq_frame_id(csv_glob)
    n_draw = 0
    
    for jf in sorted(glob.glob(events_glob)):
        seq = Path(jf).stem
        blob = json.loads(Path(jf).read_text())
        E = blob if isinstance(blob, list) else blob.get("events", [])

        for e in E:
            if e.get("type") not in ("shot", "layup"): continue
            fr = e.get("t_release")
            pid = e.get("shooter_id")
            pos = idx.get((seq, fr, pid))
            if not pos: continue
            
            u, v = ft_to_px(pos[0], pos[1], Wpx, H, pad)
            cv2.drawMarker(out, (u, v), color, markerType=cv2.MARKER_CROSS, 
                           markerSize=25, thickness=4, line_type=cv2.LINE_AA)
            n_draw += 1

    print(f"[shots] drew {n_draw} shots")
    if alpha >= 1.0: return out
    return cv2.addWeighted(out, alpha, bg, 1.0-alpha, 0)

def draw_dribbles(bg, pad, csv_glob, events_glob, color=(255, 0, 255), alpha=1.0):
    """Draw paths for dribbles (Magenta)."""
    H, Wpx = bg.shape[:2]
    out = bg.copy()
    idx = index_points_by_seq_frame_id(csv_glob)
    n_draw = 0
    
    for jf in sorted(glob.glob(events_glob)):
        seq = Path(jf).stem
        blob = json.loads(Path(jf).read_text())
        E = blob if isinstance(blob, list) else blob.get("events", [])

        for e in E:
            if e.get("type") != "dribble": continue
            f0, f1 = e.get("t_start"), e.get("t_end")
            pid = e.get("player_id")
            
            pts_px = []
            for fr in range(f0, f1 + 1):
                pos = idx.get((seq, fr, pid))
                if pos:
                    u, v = ft_to_px(pos[0], pos[1], Wpx, H, pad)
                    pts_px.append([u, v])
            
            if len(pts_px) > 1:
                pts_arr = np.array([pts_px], dtype=np.int32)
                cv2.polylines(out, pts_arr, isClosed=False, color=color, thickness=4, lineType=cv2.LINE_AA)
                n_draw += 1

    print(f"[dribbles] drew {n_draw} dribble paths")
    if alpha >= 1.0: return out
    return cv2.addWeighted(out, alpha, bg, 1.0-alpha, 0)

def draw_all_events(bg, pad, csv_glob, events_glob, alpha=1.0):
    """Chain all event drawers: Dribbles -> Passes -> Shots."""
    # We chain them with alpha=1.0 to overlay them all onto one canvas,
    # then apply the user's alpha at the very end against the ORIGINAL bg.
    img = bg.copy()
    img = draw_dribbles(img, pad, csv_glob, events_glob, alpha=1.0)
    img = draw_passes(img, pad, csv_glob, events_glob, alpha=1.0)
    img = draw_shots(img, pad, csv_glob, events_glob, alpha=1.0)
    
    if alpha >= 1.0: return img
    return cv2.addWeighted(img, alpha, bg, 1.0-alpha, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--court_png", required=True)
    ap.add_argument("--csv_glob",  required=True, help="e.g. 'results/court_tracks/val/v1_p*.csv'")
    ap.add_argument("--events_glob", default=None, help="needed for --plot passes/shots/dribbles")
    ap.add_argument("--plot", choices=["scatter","heatmap","passes", "shots", "dribbles", "all"], required=True)
    ap.add_argument("--Class", choices=["player","ball","both"], default="both")
    ap.add_argument("--team_json", default=None, help="tid->'home'/'away' for player coloring")
    ap.add_argument("--sigma", type=int, default=24, help="heatmap blur (px)")
    ap.add_argument("--pad", type=int, default=60, help="MUST equal the pad used when creating the court PNG")
    ap.add_argument("--alpha", type=float, default=1.0, help="overlay strength (1.0 = no wash-out)")
    ap.add_argument("--r_player", type=int, default=7)
    ap.add_argument("--r_ball",   type=int, default=10)
    ap.add_argument("--edge",     type=int, default=2)
    ap.add_argument("--no_clip", action="store_true", help="draw points even if outside court bounds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bg = cv2.imread(args.court_png, cv2.IMREAD_COLOR)
    assert bg is not None, f"cannot read {args.court_png}"

    teams = None
    if args.team_json and Path(args.team_json).exists():
        try:
            teams = json.loads(Path(args.team_json).read_text())
        except Exception:
            teams = None

    clip = not args.no_clip

    if args.plot == "scatter":
        want = None if args.Class == "both" else (0 if args.Class == "player" else 1)
        pts, clipped = load_csv_points(args.csv_glob, want_cls=want, clip_out=clip)
        if clip and clipped:
            print(f"[warn] clipped {clipped} points outside court bounds.")
        out = draw_scatter(bg, args.pad, pts, which=args.Class,
                           teams=teams, alpha=args.alpha,
                           r_player=args.r_player, r_ball=args.r_ball, edge_th=args.edge)
    elif args.plot == "heatmap":
        which_cls = 0 if args.Class in ("player","both") else 1
        pts, clipped = load_csv_points(args.csv_glob, want_cls=which_cls, clip_out=clip)
        if clip and clipped:
            print(f"[warn] clipped {clipped} points outside court bounds.")
        out = draw_heatmap(bg, args.pad, pts, sigma_px=args.sigma, which_cls=which_cls)
    elif args.plot == "passes":
        assert args.events_glob, "--events_glob is required"
        out = draw_passes(bg, args.pad, args.csv_glob, args.events_glob, alpha=args.alpha)
    elif args.plot == "shots":
        assert args.events_glob, "--events_glob is required"
        out = draw_shots(bg, args.pad, args.csv_glob, args.events_glob, alpha=args.alpha)
    elif args.plot == "dribbles":
        assert args.events_glob, "--events_glob is required"
        out = draw_dribbles(bg, args.pad, args.csv_glob, args.events_glob, alpha=args.alpha)
    elif args.plot == "all":
        assert args.events_glob, "--events_glob is required"
        out = draw_all_events(bg, args.pad, args.csv_glob, args.events_glob, alpha=args.alpha)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()