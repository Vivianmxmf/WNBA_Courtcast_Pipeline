#!/usr/bin/env python3
"""
Estimate image->court homographies from CVAT XML clicks.

Usage:
  python src/court/estimate_homography.py \
    --xml data/meta/cvat_exports/homography/game1.xml \
    --game_id game1 \
    --out_dir data/meta/homography \
    --seg_halves 1:left 2:right
"""
import argparse, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2

# ---------- Court model (NBA, feet) ----------
COURT_LENGTH = 94.0
COURT_WIDTH  = 50.0
HALF_LEN     = COURT_LENGTH / 2.0       # 47
HALF_WID     = COURT_WIDTH  / 2.0       # 25
HOOP_OFFSET  = 4.0                      # rim center is 4 ft in from baseline
HOOP_X_R     =  HALF_LEN - HOOP_OFFSET  # +43
HOOP_X_L     = -HALF_LEN + HOOP_OFFSET  # -43
LANE_HALF_W  = 8.0                      # key half-width (16 ft wide)
FT_LINE_FROM_BASE = 19.0                # free-throw line 19 ft from baseline

THREE_PT_RADIUS = 23.75                 # 3-pt arc radius (NBA), feet
CENTER_C_R      = 6.0                   # center-circle radius

def _corner_three_y(base_x: float, hoop_x: float) -> float:
    """y >= 0 where 3-pt circle (center=hoop, R=23.75) meets the baseline x=base_x."""
    dx = base_x - hoop_x
    val = THREE_PT_RADIUS**2 - dx**2
    return float(np.sqrt(max(0.0, val)))  # ≈ 23.42 ft

def template_points_for_half(half: str):
    """
    Return {name: (x, y)} in FEET for a given half: 'left' or 'right'.
    Supports: baseline corners, FT corners, center-circle L/R, and
              left_3pt_sideline/right_3pt_sideline (3-pt arc ∩ baseline).
    """
    if half not in ("left", "right"):
        raise ValueError("half must be 'left' or 'right'")

    if half == "left":
        base_x = -HALF_LEN              # -47
        hoop_x = HOOP_X_L               # -43
        ft_x   = base_x + FT_LINE_FROM_BASE   # -28
    else:
        base_x = +HALF_LEN              # +47
        hoop_x = HOOP_X_R               # +43
        ft_x   = base_x - FT_LINE_FROM_BASE   # +28

    y_corner3 = _corner_three_y(base_x, hoop_x)  # ≈ 23.42

    return {
        # Baseline corners (baseline ∩ sideline)
        "left_baseline_corner"  : (base_x, -HALF_WID),
        "right_baseline_corner" : (base_x, +HALF_WID),

        # Free-throw lane top corners
        "left_ft_corner"        : (ft_x,   -LANE_HALF_W),
        "right_ft_corner"       : (ft_x,   +LANE_HALF_W),

        # Center circle points (useful if visible)
        "center_circle_left"    : (-CENTER_C_R, 0.0),
        "center_circle_right"   : (+CENTER_C_R, 0.0),

        # NEW: 3-pt arc meets baseline (two symmetric points)
        "left_3pt_sideline"     : (base_x, -y_corner3),
        "right_3pt_sideline"    : (base_x, +y_corner3),
    }

# ---------- Robust CVAT XML parsing ----------
_ATTR_KEYS_FOR_NAME = ("kind", "name")

def _attr_text(a):
    return a.get("value") or (a.text.strip() if a.text else None)

def _extract_point_name(attr_elems):
    for a in attr_elems:
        if a.get("name") in _ATTR_KEYS_FOR_NAME:
            val = _attr_text(a)
            if val:
                return val
    return None

def parse_cvat_points(xml_path: Path, label_name="court_point"):
    """
    Returns dict: seg_id -> { clicked_name: (u, v) } in IMAGE PIXELS.
    Works for CVAT-for-Images and CVAT-for-Video exports.
    Accepts point-name in attribute 'kind' or 'name' (value=... or inner text).
    """
    root = ET.parse(xml_path).getroot()
    segmap = {}
    found_any = False

    # A) Images
    for img in root.findall(".//image"):
        for pt in img.findall("./points"):
            if pt.attrib.get("label") != label_name:
                continue
            uv = pt.attrib.get("points")
            if not uv:
                continue
            try:
                u, v = [float(x) for x in uv.split(",")[:2]]
            except Exception:
                continue
            name_val = _extract_point_name(pt.findall("./attribute")) or (pt.attrib.get("name") or "").strip()
            if not name_val:
                continue
            seg_id = 1
            for a in pt.findall("./attribute"):
                if a.get("name") == "seg_id":
                    try:
                        seg_id = int(float(_attr_text(a)))
                    except Exception:
                        seg_id = 1
            segmap.setdefault(seg_id, {})[name_val] = (u, v)
            found_any = True

    # B) Video (tracks)
    for tr in root.findall(f".//track[@label='{label_name}']"):
        shape = tr.find("./points")
        if shape is None:
            continue
        uv = shape.attrib.get("points")
        if not uv:
            continue
        try:
            u, v = [float(x) for x in uv.split(",")[:2]]
        except Exception:
            continue
        name_val = _extract_point_name(shape.findall("./attribute")) or (shape.attrib.get("name") or "").strip()
        if not name_val:
            continue
        seg_id = 1
        for a in shape.findall("./attribute"):
            if a.get("name") == "seg_id":
                try:
                    seg_id = int(float(_attr_text(a)))
                except Exception:
                    seg_id = 1
        segmap.setdefault(seg_id, {})[name_val] = (u, v)
        found_any = True

    if not found_any:
        print(f"[warn] No '{label_name}' points parsed from {xml_path}.")
    return segmap

def estimate_h(pts_img, pts_world):
    """Compute H with RANSAC. pts_img/world are Nx2 arrays."""
    if len(pts_img) < 4:
        return None, None
    H, mask = cv2.findHomography(pts_world, pts_img, cv2.RANSAC, ransacReprojThreshold=3.0)
    return H, mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="CVAT XML path")
    ap.add_argument("--game_id", required=True, help="e.g., game1")
    ap.add_argument("--out_dir", default="data/meta/homography")
    ap.add_argument("--seg_halves", nargs="*", default=["1:left","2:right"],
                    help="Map seg_id to half, e.g., 1:left 2:right")
    args = ap.parse_args()

    seg_half_map = {}
    for token in args.seg_halves:
        k, v = token.split(":")
        seg_half_map[int(k)] = v

    xml_path = Path(args.xml)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_points = parse_cvat_points(xml_path)
    if not seg_points:
        print(f"[err] Parsed 0 segments from {xml_path}. Nothing to write.")
        return

    # Normalize common aliases from your CVAT menu
    name_alias = {
        "left_lane_top":  "left_ft_corner",
        "right_lane_top": "right_ft_corner",
        # keep 3pt names as-is
    }

    wrote_any = False
    for seg_id, clicked in seg_points.items():
        clicked_norm = {name_alias.get(k, k): uv for k, uv in clicked.items()}

        half = seg_half_map.get(seg_id, "left")
        template = template_points_for_half(half)

        common = sorted(set(clicked_norm) & set(template))
        if len(common) < 4:
            print(f"[warn] seg {seg_id} ({half}): only {len(common)} usable points: {common}. Need ≥4; skipping.")
            continue

        pts_img   = np.array([clicked_norm[n] for n in common], dtype=np.float32)
        pts_world = np.array([template[n]      for n in common], dtype=np.float32)
        H, mask   = estimate_h(pts_img, pts_world)

        out = {
            "game_id": args.game_id,
            "seg_id": int(seg_id),
            "half": half,
            "src_xml": str(xml_path),
            "used_points": common,
            "points_image": {n: list(map(float, clicked_norm[n])) for n in common},
            "points_court": {n: list(map(float, template[n])) for n in common},
            "H": (H.tolist() if H is not None else None),
            "inliers": (mask.flatten().astype(int).tolist() if mask is not None else None),
            "units": "feet",
            "model": "NBA half-court (3pt corner via circle∩baseline)",
        }

        out_name = f"{args.game_id}_seg{seg_id}.json"
        (out_dir / out_name).write_text(json.dumps(out, indent=2))
        print(f"[OK] wrote {out_dir/out_name}  (points: {len(common)}, "
              f"inliers: {int(mask.sum()) if mask is not None else 0})")
        wrote_any = True

    if not wrote_any:
        print("[err] No homographies estimated. Label ≥4 of the supported points and re-run.")

if __name__ == "__main__":
    main()