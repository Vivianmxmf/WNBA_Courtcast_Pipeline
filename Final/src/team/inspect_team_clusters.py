#!/usr/bin/env python3
"""
Visual inspection for jersey clustering.

Reads the CSV reports written by color_cluster.py (kmeans / gmm / prototypes),
builds margin plots, scatter plots (home vs away metric), and exports an
"ambiguous" shortlist (lowest margins) to CSV.

Usage (defaults assume game 'v1' and your current paths):
  python3 src/team/inspect_team_clusters.py --game v1
  # or specify explicit CSVs:
  python3 src/team/inspect_team_clusters.py \
      --game v1 \
      --kmeans_csv data/meta/team_assign/v1_margin_kmeans.csv \
      --gmm_csv    data/meta/team_assign/v1_margin_gmm.csv \
      --proto_csv  data/meta/team_assign/v1_margin_prototypes.csv \
      --out_dir    data/meta/team_assign/plots_v1

Notes
- The script is tolerant to column names. It looks for:
    tid / track_id / id
    assign / label / pred
    margin
    *_home, *_away  (e.g., dist_home/dist_away, prob_home/prob_away, sim_home/sim_away)
- If 'margin' is missing it computes |home - away|.
- “Metric” semantics:
    * for KMeans/Prototypes, HOME/AWAY columns are typically distances (smaller is better),
      but margin is still |home - away| so larger margin = more confident.
    * for GMM, HOME/AWAY are usually probabilities (0..1); same margin logic applies.
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def _colmap_lower(df: pd.DataFrame):
    """lowercase column mapping: lower -> original"""
    return {c.lower(): c for c in df.columns}

def _pick(cols_lc: dict, *candidates):
    """return original-name for the first candidate (lowercase match or substring '*home'/'*away')."""
    for cand in candidates:
        if cand in cols_lc:
            return cols_lc[cand]
    # fuzzy: try any column that endswith candidate
    for cand in candidates:
        for lc, orig in cols_lc.items():
            if lc.endswith(cand):
                return orig
    return None

def _find_home_away(df: pd.DataFrame):
    """Try to find a pair of HOME/AWAY metric columns in df (distance or probability)."""
    lc = _colmap_lower(df)
    # common patterns
    pairs = [
        ("dist_home", "dist_away"),
        ("prob_home", "prob_away"),
        ("p_home", "p_away"),
        ("score_home", "score_away"),
        ("sim_home", "sim_away"),
        ("d_home", "d_away"),
        ("home", "away"),
    ]
    for h, a in pairs:
        ch = _pick(lc, h)
        ca = _pick(lc, a)
        if ch and ca:
            return ch, ca
    # last resort: any two columns containing 'home'/'away'
    home = [orig for lc,orig in lc.items() if "home" in lc]
    away = [orig for lc,orig in lc.items() if "away" in lc]
    if home and away:
        return home[0], away[0]
    return None, None

def load_csv_generic(path: Path):
    """Load CSV and normalize to columns: tid, assign, margin, home_metric, away_metric."""
    df = pd.read_csv(path)
    lc = _colmap_lower(df)

    tid_col = _pick(lc, "tid", "track_id", "id")
    if not tid_col:
        raise ValueError(f"{path}: could not find TID column (tid/track_id/id).")

    assign_col = _pick(lc, "assign", "label", "pred")
    margin_col = _pick(lc, "margin")

    home_col, away_col = _find_home_away(df)

    out = pd.DataFrame()
    out["tid"] = df[tid_col].astype(str)

    if assign_col:
        out["assign"] = df[assign_col].astype(str)
    else:
        out["assign"] = ""  # optional

    if margin_col:
        out["margin"] = pd.to_numeric(df[margin_col], errors="coerce")
    else:
        if home_col and away_col:
            home_vals = pd.to_numeric(df[home_col], errors="coerce")
            away_vals = pd.to_numeric(df[away_col], errors="coerce")
            out["margin"] = (home_vals - away_vals).abs()
        else:
            # fallback: all zero margins; not ideal but keeps script running
            out["margin"] = 0.0

    if home_col: out["home_metric"] = pd.to_numeric(df[home_col], errors="coerce")
    else:        out["home_metric"] = np.nan
    if away_col: out["away_metric"] = pd.to_numeric(df[away_col], errors="coerce")
    else:        out["away_metric"] = np.nan

    # drop rows that are entirely NaN margin (rare)
    out = out[~out["margin"].isna()].reset_index(drop=True)
    return out

def plot_sorted_margins(df: pd.DataFrame, algo: str, out_dir: Path, top_k: int = 30):
    d = df.sort_values("margin", ascending=True).head(min(top_k, len(df)))
    plt.figure(figsize=(10, max(4, 0.35*len(d))))
    plt.barh(d["tid"].astype(str), d["margin"].values)
    plt.xlabel("margin (|home − away|)")
    plt.ylabel("track id (lowest margins first)")
    plt.title(f"{algo.upper()} — lowest margins")
    plt.tight_layout()
    p = out_dir / f"{algo}_sorted_margins.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return p

def plot_hist(df: pd.DataFrame, algo: str, out_dir: Path, bins: int = 20):
    q = df["margin"].quantile([0.25, 0.5, 0.75]).values
    plt.figure(figsize=(7,4))
    plt.hist(df["margin"].values, bins=bins)
    for val, ls in zip(q, ["--", "-", "--"]):
        plt.axvline(val, color="k", linestyle=ls, alpha=0.7)
    plt.xlabel("margin")
    plt.ylabel("count")
    plt.title(f"{algo.upper()} — margin histogram (Q1/Q2/Q3 marked)")
    plt.tight_layout()
    p = out_dir / f"{algo}_margin_hist.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return p

def plot_scatter(df: pd.DataFrame, algo: str, out_dir: Path):
    if not {"home_metric","away_metric"}.issubset(df.columns):
        return None
    plt.figure(figsize=(5.5,5.5))
    x = df["home_metric"].values
    y = df["away_metric"].values
    c = (df["assign"].astype(str) == "home").map({True:0, False:1}).values
    # simple coloring without seaborn
    plt.scatter(x, y, s=18, c=c, cmap="tab10", alpha=0.8)
    mn = np.nanmin([x.min(), y.min()])
    mx = np.nanmax([x.max(), y.max()])
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    plt.xlabel("home metric")
    plt.ylabel("away metric")
    plt.title(f"{algo.upper()} — home vs away metric")
    plt.tight_layout()
    p = out_dir / f"{algo}_home_vs_away.png"
    plt.savefig(p, dpi=150)
    plt.close()
    return p

def summarize_margins(df: pd.DataFrame):
    s = df["margin"].describe(percentiles=[0.25,0.5,0.75]).to_dict()
    return {k: float(v) for k,v in s.items()}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="game id (used for defaults)")
    ap.add_argument("--kmeans_csv", default=None)
    ap.add_argument("--gmm_csv",    default=None)
    ap.add_argument("--proto_csv",  default=None)
    ap.add_argument("--out_dir",    default=None)
    ap.add_argument("--top_k", type=int, default=30, help="# of lowest-margin IDs to show in bar chart")
    ap.add_argument("--ambig_percent", type=float, default=25.0, help="bottom P% margins exported to CSV")
    args = ap.parse_args()

    game = args.game
    kmeans_csv = Path(args.kmeans_csv or f"data/meta/team_assign/{game}_margin_kmeans.csv")
    gmm_csv    = Path(args.gmm_csv    or f"data/meta/team_assign/{game}_margin_gmm.csv")
    proto_csv  = Path(args.proto_csv  or f"data/meta/team_assign/{game}_margin_prototypes.csv")
    out_dir    = Path(args.out_dir or f"data/meta/team_assign/plots_{game}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for algo, csv_path in [("kmeans", kmeans_csv), ("gmm", gmm_csv), ("prototypes", proto_csv)]:
        if not csv_path.exists():
            print(f"[WARN] missing CSV for {algo}: {csv_path}")
            continue
        df = load_csv_generic(csv_path)
        if df.empty:
            print(f"[WARN] empty data for {algo}: {csv_path}")
            continue

        # plots
        p1 = plot_sorted_margins(df, algo, out_dir, args.top_k)
        p2 = plot_hist(df, algo, out_dir)
        p3 = plot_scatter(df, algo, out_dir)

        # ambiguous shortlist
        cutoff = np.percentile(df["margin"].values, args.ambig_percent)
        ambig = df[df["margin"] <= cutoff].sort_values("margin")
        ambig_out = out_dir / f"ambiguous_{algo}.csv"
        ambig.to_csv(ambig_out, index=False)

        # summary
        stats = summarize_margins(df)
        results[algo] = {
            "csv": str(csv_path),
            "plots": {
                "sorted_margins": (str(p1) if p1 else None),
                "hist": (str(p2) if p2 else None),
                "home_vs_away": (str(p3) if p3 else None),
            },
            "ambiguous_csv": str(ambig_out),
            "margin_summary": stats,
            "n_tids": int(len(df)),
        }
        print(f"[{algo}] n={len(df)}  margin min/med/max = "
              f"{df['margin'].min():.3f}/{df['margin'].median():.3f}/{df['margin'].max():.3f}")
        print(f"[{algo}] wrote: {ambig_out}")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[DONE] visual inspection written to {out_dir}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()