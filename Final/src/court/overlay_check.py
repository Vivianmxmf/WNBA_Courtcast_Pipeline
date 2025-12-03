#!/usr/bin/env python3
"""
Overlay a half-court template via a homography onto an image for quick QA.

Usage:
  python src/court/overlay_check.py \
    --image data/dataset/images/val_fixed/v1_p7_00045.png \
    --H data/meta/homography/game1_seg1.json \
    --out overlay_game1_seg1.png
"""
import argparse, json
from pathlib import Path
import numpy as np, cv2

# Feet (same constants as estimate_homography)
COURT_LENGTH = 94.0
COURT_WIDTH  = 50.0
HALF_LEN     = COURT_LENGTH / 2.0
HALF_WID     = COURT_WIDTH  / 2.0
HOOP_OFFSET  = 4.0
HOOP_X_R     =  HALF_LEN - HOOP_OFFSET
HOOP_X_L     = -HALF_LEN + HOOP_OFFSET
LANE_HALF_W  = 8.0
FT_LINE_FROM_BASE = 19.0
TOP_ARC_DIST = 23.75

def half_polylines(half: str, n_arc=120):
    """Return list of polylines (each Nx2 in feet) for half-court."""
    lines = []
    if half == "left":
        base_x = -HALF_LEN
        hoop_x = HOOP_X_L
        ft_x   = base_x + FT_LINE_FROM_BASE
        top_arc_x = hoop_x + TOP_ARC_DIST
    else:
        base_x = +HALF_LEN
        hoop_x = HOOP_X_R
        ft_x   = base_x - FT_LINE_FROM_BASE
        top_arc_x = hoop_x - TOP_ARC_DIST

    # Sidelines & baseline (visible half)
    lines.append(np.array([[base_x, -HALF_WID], [base_x, HALF_WID]], float))   # baseline
    lines.append(np.array([[base_x, -HALF_WID], [0.0,   -HALF_WID]], float))   # lower sideline
    lines.append(np.array([[base_x,  HALF_WID], [0.0,    HALF_WID]], float))   # upper sideline

    # Key (lane)
    lines.append(np.array([[base_x, -LANE_HALF_W], [ft_x, -LANE_HALF_W]], float))
    lines.append(np.array([[base_x,  LANE_HALF_W], [ft_x,  LANE_HALF_W]], float))
    lines.append(np.array([[ft_x,   -LANE_HALF_W], [ft_x,  LANE_HALF_W]], float))

    # Free-throw semicircle (approximate by polyline)
    # radius 6 ft around (ft_x, 0), only inside the half
    t = np.linspace(-np.pi/2, np.pi/2, 90)  # front half
    arc = np.stack([ft_x + 6*np.cos(t), 0 + 6*np.sin(t)], axis=1)
    if half == "left":
        arc = arc[arc[:,0] >= base_x]  # keep inside
    else:
        arc = arc[arc[:,0] <= base_x]  # not really needed
    lines.append(arc)

    # 3-point arc around hoop
    # Arc of radius 23.75 ft centered at hoop (clip to half)
    ang = np.linspace(-np.pi/2, np.pi/2, n_arc)
    arc3 = np.stack([hoop_x + TOP_ARC_DIST*np.cos(ang), 0 + TOP_ARC_DIST*np.sin(ang)], axis=1)
    if half == "left":
        arc3 = arc3[arc3[:,0] >= base_x]
    else:
        arc3 = arc3[arc3[:,0] <= base_x]
    lines.append(arc3)

    return lines

def warp_points(H, pts):
    pts_h = np.concatenate([pts, np.ones((pts.shape[0],1))], axis=1)  # Nx3
    img_h = (H @ pts_h.T).T
    img_h[:,0] /= img_h[:,2]
    img_h[:,1] /= img_h[:,2]
    return img_h[:,:2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--H", required=True)
    ap.add_argument("--out", default="overlay.png")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    meta = json.loads(Path(args.H).read_text())
    H = np.array(meta["H"], float)
    half = meta.get("half","left")

    lines = half_polylines(half)
    disp = img.copy()
    for poly in lines:
        W = warp_points(H, poly)
        W = np.round(W).astype(int)
        # draw
        for i in range(len(W)-1):
            p1, p2 = tuple(W[i]), tuple(W[i+1])
            # skip if wildly outside
            if not (abs(p1[0])>1e6 or abs(p1[1])>1e6 or abs(p2[0])>1e6 or abs(p2[1])>1e6):
                cv2.line(disp, p1, p2, (0,255,0), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, disp)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()