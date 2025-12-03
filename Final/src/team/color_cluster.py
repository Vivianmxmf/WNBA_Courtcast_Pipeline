#!/usr/bin/env python3
"""
Unsupervised jersey clustering (K=2) with optional seeds or fixed prototypes.

Inputs:
  - data/meta/team_crops/<game>/<tid>/*.png
Optional:
  - data/meta/team_assign/<game>_seeds.json   # {"tid": "home"/"away", ...}
  - --jersey_home, --jersey_away              # two jersey images as prototypes
Outputs:
  - data/meta/team_assign/<game>.json         # {"tid": "home"/"away", ...}
  - --report_csv (optional)                   # tid, dist_home, dist_away, margin, predicted
"""
import argparse, json, csv
from pathlib import Path
import numpy as np, cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from collections import Counter

# -------- features --------
def crop_feature(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    # tighter central torso patch (helps reduce shorts/court spill)
    y0, y1 = int(0.30 * H), int(0.70 * H)
    x0, x1 = int(0.30 * W), int(0.70 * W)
    roi = hsv[y0:y1, x0:x1]
    flat = roi.reshape(-1, 3)
    feat = np.concatenate([flat.mean(0), np.median(flat, axis=0)])
    return feat.astype(np.float32)

def feature_from_image(img_path: str):
    # Convenience wrapper to keep prototype/seed paths compatible with crop_feature.
    return crop_feature(img_path)

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="folder under data/meta/team_crops")
    ap.add_argument("--crops_root", default="data/meta/team_crops")
    ap.add_argument("--out_root",   default="data/meta/team_assign")
    ap.add_argument("--seeds", default=None, help="optional seeds json")
    ap.add_argument("--algo", choices=["kmeans", "gmm"], default="kmeans",
                    help="clustering algorithm (ignored if prototypes given)")
    ap.add_argument("--jersey_home", default=None, help="path to home jersey image (prototype mode)")
    ap.add_argument("--jersey_away", default=None, help="path to away jersey image (prototype mode)")
    ap.add_argument("--report_csv", default=None, help="optional CSV path with per-TID distances/margins")
    args = ap.parse_args()

    game_dir = Path(args.crops_root) / args.game
    assert game_dir.exists(), f"missing crops: {game_dir}"
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # per-TID features (mean over its crops)
    feats, tids = [], []
    for tid_dir in sorted([d for d in game_dir.iterdir() if d.is_dir()]):
        fs = []
        for p in tid_dir.glob("*.png"):
            f = crop_feature(p)
            if f is not None:
                fs.append(f)
        if not fs:
            continue
        feats.append(np.mean(np.stack(fs, 0), 0))
        tids.append(tid_dir.name)
    if not feats:
        raise SystemExit("No features found.")
    X = np.stack(feats, 0)

    used_prototypes = args.jersey_home and args.jersey_away
    dh = da = None  # distances (for CSV reporting)

    # --- labeling ---
    if used_prototypes:
        ph = feature_from_image(args.jersey_home)
        pa = feature_from_image(args.jersey_away)
        if ph is None or pa is None:
            raise SystemExit("Failed to read jersey prototype image(s).")
        dh = np.linalg.norm(X - ph, axis=1)   # distance to home prototype
        da = np.linalg.norm(X - pa, axis=1)   # distance to away prototype
        # label: 0=home (closer to home prototype), 1=away
        y = (da < dh).astype(int)
        algo_used = "prototypes"
    else:
        if args.algo == "kmeans":
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            y = km.fit_predict(X)
            algo_used = "kmeans"
        else:
            gm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
            y = gm.fit_predict(X)
            algo_used = "gmm"

    # silhouette (only meaningful when both clusters non-empty)
    try:
        if len(set(y)) == 2:
            sil = silhouette_score(X, y)
            print(f"[report] silhouette={sil:.3f} ({algo_used})")
        else:
            print(f"[report] single cluster detected by {algo_used}; silhouette skipped.")
    except Exception as _:
        pass

    # default mapping: cluster 0 -> home, cluster 1 -> away
    mapping = {0: "home", 1: "away"}

    # If prototypes were used, mapping is fixed; otherwise allow seeds to flip mapping if needed
    if (not used_prototypes) and args.seeds and Path(args.seeds).exists():
        seeds = json.loads(Path(args.seeds).read_text())
        votes = Counter()
        for tid, lbl in seeds.items():
            if tid in tids:
                ci = y[tids.index(tid)]
                votes[(ci, lbl)] += 1
        decided = {}
        for ci in [0, 1]:
            options = [(k[1], v) for k, v in votes.items() if k[0] == ci]
            if options:
                decided[ci] = max(options, key=lambda z: z[1])[0]
        if len(decided) == 2 and len(set(decided.values())) == 2:
            mapping = {0: decided[0], 1: decided[1]}
        elif len(decided) == 1:
            # ensure distinct mapping
            only_ci = list(decided.keys())[0]
            other_lbl = "away" if decided[only_ci] == "home" else "home"
            mapping = {only_ci: decided[only_ci], 1 - only_ci: other_lbl}

    assign = {tids[i]: mapping[int(y[i])] for i in range(len(tids))}
    out_path = out_dir / f"{args.game}.json"
    out_path.write_text(json.dumps(assign, indent=2))
    print(f"[OK] wrote {out_path}")

    # CSV sanity report (optional)
    if args.report_csv:
        rows = []
        if dh is None or da is None:
            # approximate distances via cluster means to emulate prototype distances
            if len(set(y)) == 2:
                c0 = X[y == 0].mean(0)
                c1 = X[y == 1].mean(0)
                dh = np.linalg.norm(X - c0, axis=1)  # "home-like"
                da = np.linalg.norm(X - c1, axis=1)  # "away-like"
            else:
                # fall back to zeros to avoid crash; margins will be 0
                dh = np.zeros(len(tids), dtype=float)
                da = np.zeros(len(tids), dtype=float)
        for i, tid in enumerate(tids):
            margin = float(abs(dh[i] - da[i]))
            rows.append([tid, float(dh[i]), float(da[i]), margin, assign[tid]])
        rp = Path(args.report_csv)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tid", "dist_home", "dist_away", "margin", "predicted"])
            w.writerows(rows)
        print(f"[OK] wrote CSV sanity: {rp}")

if __name__ == "__main__":
    main()
