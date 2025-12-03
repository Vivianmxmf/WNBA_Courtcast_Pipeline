#!/usr/bin/env python3
"""
fine_tune_from_owner_csv.py

Use owner-distance diagnostics to propose robust thresholds:
- own_px      : distance threshold (px) to attach ball to nearest player
- own_hys     : hysteresis band for ownership stickiness (px)
- near_px     : 'near' band for dribble oscillation (px)
- far_px      : 'far' band (px), must be > near_px

Inputs:  one or more CSVs produced by your owner-distance logger.
Expected columns (flexible, we auto-detect by name):
  * frame (optional)
  * owner_id or owner (optional)
  * dist_owner_px  (or: owner_dist_px, d_owner_px, dist_owner)
  * dist_min_px    (or: min_dist_px, nearest_dist_px, dist_min, nearest_px)

Outputs:
  * YAML with suggested thresholds
  * summary CSV with per-seq percentiles
  * histograms with thresholds overlaid

Usage:
  python3 src/tune/fine_tune_from_owner_csv.py \
    --csv_glob "results/analysis/owner_dist/val/*_owner_dist.csv" \
    --out_yaml results/tune/rules_suggest.yaml \
    --plots_dir results/tune/owner_plots \
    --summary_csv results/tune/owner_thresholds_summary.csv \
    --own_pct 0.85 --near_pct 0.25 --far_pct 0.75 \
    --hys_frac 0.15

Tips:
- If your CSVs live elsewhere, just adjust --csv_glob.
- For 2 FPS clips, you will still tune frame-based rules in rules_actions.py
  (pass_air / shot_air) separately; this script only suggests pixel thresholds.
"""
import argparse, glob, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df, candidates):
    """
    Return the first matching column name in df for any of the provided
    lowercase candidate patterns. Matches exact lowercase first, then
    substring on lowercase.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    # exact
    for pat in candidates:
        if pat in cols_lower:
            return cols_lower[pat]
    # substring
    for pat in candidates:
        for lc, orig in cols_lower.items():
            if pat in lc:
                return orig
    return None


def load_one_csv(fp):
    """
    Robustly load a single owner-distance CSV.

    We accept your current schema:
      - dist_px : distance between ball and current owner (pixels)
      - ball_px, ball_py, own_px, own_py : raw image coords
      - dist_ft : same distance but in feet (not needed if we can compute px)

    Returns:
      owner_dist_px : np.ndarray (ball-to-owner distance in px)
      min_dist_px   : np.ndarray (proxy for nearest-player distance in px)
                      If not provided, we fall back to owner distance.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(fp)

    # 1) Try to grab a direct "owner distance in px".
    #    Broadened synonym list to include your 'dist_px'.
    owner_col = pick_col(df, [
        "dist_owner_px", "owner_dist_px", "d_owner_px",
        "dist_owner", "owner_dist", "dist_px"  # <-- your column
    ])

    # 2) Try to grab a true "min distance in px" if present (usually absent).
    min_col = pick_col(df, [
        "dist_min_px", "min_dist_px", "nearest_dist_px",
        "dist_min", "nearest_px", "dmin"
    ])

    # 3) If no direct dist_px but we have raw positions, compute dist_px.
    if owner_col is None:
        bx = pick_col(df, ["ball_px", "ballx", "x_ball"])
        by = pick_col(df, ["ball_py", "bally", "y_ball"])
        ox = pick_col(df, ["own_px", "owner_px", "x_owner"])
        oy = pick_col(df, ["own_py", "owner_py", "y_owner"])
        if all(c is not None for c in [bx, by, ox, oy]):
            bxp = df[bx].astype(float).to_numpy()
            byp = df[by].astype(float).to_numpy()
            oxp = df[ox].astype(float).to_numpy()
            oyp = df[oy].astype(float).to_numpy()
            owner_dist_px = np.sqrt((bxp - oxp)**2 + (byp - oyp)**2)
        else:
            # last resort: do we have distance in feet only?
            # (We avoid ft→px conversion guesses; better to skip.)
            dft = pick_col(df, ["dist_ft", "distance_ft"])
            if dft is not None:
                # Use feet distances directly as a proxy (will still give
                # sane percentiles; we only use them to *rank* and split bands).
                owner_dist_px = df[dft].astype(float).to_numpy()
            else:
                raise ValueError(
                    f"{fp}: could not derive owner distance; "
                    f"need either dist_px OR (ball_px,ball_py,own_px,own_py) OR dist_ft."
                )
    else:
        owner_dist_px = df[owner_col].astype(float).to_numpy()

    # 4) If min distance is absent, fall back to owner distance (proxy).
    if min_col is None:
        min_dist_px = owner_dist_px.copy()
    else:
        min_dist_px = df[min_col].astype(float).to_numpy()

    # 5) Clean NaNs/Infs
    owner_dist_px = pd.Series(owner_dist_px).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    min_dist_px   = pd.Series(min_dist_px).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    if owner_dist_px.size == 0 or min_dist_px.size == 0:
        raise ValueError(f"{fp}: empty numeric distances after cleaning.")

    return owner_dist_px, min_dist_px

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", required=True)
    ap.add_argument("--out_yaml", required=True)
    ap.add_argument("--summary_csv", default=None)
    ap.add_argument("--plots_dir", default=None)
    ap.add_argument("--own_pct", type=float, default=0.85, help="percentile for own_px on min distance")
    ap.add_argument("--near_pct", type=float, default=0.25, help="percentile for near_px on owner distance")
    ap.add_argument("--far_pct",  type=float, default=0.75, help="percentile for far_px on owner distance")
    ap.add_argument("--hys_frac", type=float, default=0.15, help="own_hys as fraction of own_px")
    args = ap.parse_args()

    fps_note = "This tool proposes *pixel* thresholds. Keep your frame thresholds (pass_air/shot_air) FPS-aware."

    files = sorted(glob.glob(args.csv_glob))
    if not files:
        print(f"[err] No CSVs match {args.csv_glob}", file=sys.stderr)
        sys.exit(2)

    if args.plots_dir:
        Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    all_owner = []
    all_min   = []

    print(f"[info] Found {len(files)} CSV(s). Reading…")
    for fp in files:
        try:
            owner_px, min_px = load_one_csv(fp)
        except Exception as e:
            print(f"[warn] skip {fp}: {e}", file=sys.stderr)
            continue
        if owner_px.size == 0 or min_px.size == 0:
            print(f"[warn] skip {fp}: empty columns", file=sys.stderr)
            continue

        seq = Path(fp).stem.replace("_owner_dist","")

        # per-seq percentiles
        def pct(arr, p): return float(np.nanpercentile(arr, p*100.0))
        pcts = {
            "owner_p10": pct(owner_px, 0.10),
            "owner_p25": pct(owner_px, 0.25),
            "owner_p50": pct(owner_px, 0.50),
            "owner_p75": pct(owner_px, 0.75),
            "owner_p90": pct(owner_px, 0.90),
            "min_p10":   pct(min_px,   0.10),
            "min_p25":   pct(min_px,   0.25),
            "min_p50":   pct(min_px,   0.50),
            "min_p75":   pct(min_px,   0.75),
            "min_p90":   pct(min_px,   0.90),
            "n_owner": int(owner_px.size),
            "n_min":   int(min_px.size),
        }
        rows.append({"seq": seq, **pcts})
        all_owner.append(owner_px)
        all_min.append(min_px)

        # per-seq histogram
        if args.plots_dir:
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.hist(owner_px, bins=40, alpha=0.55, label="owner_dist_px")
            ax.hist(min_px,   bins=40, alpha=0.35, label="min_dist_px")
            near_val = np.nanpercentile(owner_px, args.near_pct*100.0)
            far_val  = np.nanpercentile(owner_px, args.far_pct*100.0)
            own_val  = np.nanpercentile(min_px,   args.own_pct*100.0)
            for x,c,lw,lb in [(near_val,'#2ca02c',2,"near"),
                              (far_val,'#d62728',2,"far"),
                              (own_val,'#1f77b4',2,"own")]:
                ax.axvline(x, color=c, linewidth=lw, linestyle='--')
                ax.text(x, ax.get_ylim()[1]*0.9, lb, color=c, ha='center', va='top')
            ax.set_title(f"{seq}: distance hist (px)")
            ax.set_xlabel("pixels")
            ax.set_ylabel("count")
            ax.legend()
            out_png = Path(args.plots_dir)/f"{seq}_owner_min_hist.png"
            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)

    if not rows:
        print("[err] No usable CSVs parsed.", file=sys.stderr)
        sys.exit(3)

    df = pd.DataFrame(rows).sort_values("seq")
    if args.summary_csv:
        Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.summary_csv, index=False)
        print(f"[OK] wrote {args.summary_csv}")

    # Global suggestions (robust percentiles pooled)
    all_owner = np.concatenate(all_owner, axis=0)
    all_min   = np.concatenate(all_min,   axis=0)

    own_px  = float(np.nanpercentile(all_min,   args.own_pct*100.0))
    near_px = float(np.nanpercentile(all_owner, args.near_pct*100.0))
    far_px  = float(np.nanpercentile(all_owner, args.far_pct*100.0))

    # enforce sensible ordering and margins
    if near_px >= far_px:
        # widen bands slightly
        med = float(np.nanpercentile(all_owner, 50))
        near_px = min(med, far_px * 0.8)
    # own_px should be above far_px (and typically >= ~1.2*far)
    if own_px <= far_px:
        own_px = far_px * 1.15

    # clamp to reasonable ranges (basketball sideline cam typical)
    own_px  = float(np.clip(own_px,  80, 220))
    near_px = float(np.clip(near_px, 20, 120))
    far_px  = float(np.clip(far_px,  60, 180))

    own_hys = float(max(8.0, min(40.0, args.hys_frac * own_px)))

    # write tiny YAML (no extra deps)
    out = Path(args.out_yaml)
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml_txt = (
        "suggested:\n"
        f"  own_px: {own_px:.1f}\n"
        f"  own_hys: {own_hys:.1f}\n"
        f"  near_px: {near_px:.1f}\n"
        f"  far_px: {far_px:.1f}\n"
        "notes:\n"
        f"  - pooled_files: {len(files)}\n"
        f"  - strategy: own=min_dist P{int(args.own_pct*100)}, "
        f"near=owner_dist P{int(args.near_pct*100)}, far=owner_dist P{int(args.far_pct*100)}\n"
        f"  - {fps_note}\n"
    )
    out.write_text(yaml_txt)
    print(f"[OK] wrote {args.out_yaml}")
    print("\n--- Proposed CLI (pixel thresholds only) ---")
    print(
        f"python3 src/rules/rules_actions.py "
        f"--split val --tracker botsort --player_cls 0 --ball_cls 1 "
        f"--own_px {own_px:.1f} --own_hys {own_hys:.1f} "
        f"--near_px {near_px:.1f} --far_px {far_px:.1f} "
        f"--min_hold_f 1 --debounce_f 3 "
        f"--pass_air_min 0 --pass_air_max 6 --pass_travel 60 "
        f"--shot_air 2 --shot_speed 3"
    )

if __name__ == "__main__":
    main()