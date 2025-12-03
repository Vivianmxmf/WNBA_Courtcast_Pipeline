
#!/usr/bin/env python3
"""
Project MOT tracks (players/ball) to 2D court coordinates.
- Supports on-the-fly team classification (KMeans/GMM/Prototypes).
- SAVES the classification to data/meta/team_assign/<game>.json
- Writes CSV with 'team' column.
"""

import argparse, csv, json, re
from pathlib import Path
import numpy as np

# --- Imports for Classification ---
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

COURT_W, COURT_H = 94.0, 50.0
X_MIN, X_MAX = -COURT_W/2, COURT_W/2
Y_MIN, Y_MAX = -COURT_H/2, COURT_H/2

# ----------------- IO helpers -----------------
def load_homographies(hom_root: Path, game_id: str):
    half2inv = {}
    for p in sorted((hom_root).glob(f"{game_id}_seg*.json")):
        try:
            obj = json.loads(p.read_text())
            half = obj.get("half") or ""
            H = np.array(obj["H"], dtype=np.float64)
            if H.shape == (3,3) and half in ("left","right"):
                half2inv[half] = np.linalg.inv(H)
        except Exception:
            pass
    return half2inv

def load_mot(mot_file: Path):
    by_f = {}
    if not mot_file.exists():
        return by_f
    for line in mot_file.read_text().splitlines():
        if not line.strip(): continue
        toks = re.split(r"[,\s]+", line.strip())
        if len(toks) < 8:
            continue
        fr   = int(float(toks[0])); tid = int(float(toks[1]))
        x,y,w,h = map(float, toks[2:6])
        conf = float(toks[6])
        cls  = int(float(toks[7]))
        by_f.setdefault(fr, []).append((tid, cls, x,y,w,h, conf))
    return by_f

# ----------------- Team Classification -----------------
def crop_feature(path):
    img = cv2.imread(str(path))
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]
    y0, y1 = int(0.30 * H), int(0.70 * H)
    x0, x1 = int(0.30 * W), int(0.70 * W)
    roi = hsv[y0:y1, x0:x1]
    flat = roi.reshape(-1, 3)
    feat = np.concatenate([flat.mean(0), np.median(flat, axis=0)])
    return feat.astype(np.float32)

def compute_team_map(game_id, crops_root, algo, j_home, j_away):
    game_dir = Path(crops_root) / game_id
    if not game_dir.exists():
        print(f"[WARN] No crops at {game_dir}")
        return {}

    feats, tids = [], []
    tid_dirs = sorted([d for d in game_dir.iterdir() if d.is_dir()])
    for tid_dir in tid_dirs:
        fs = []
        for p in tid_dir.glob("*.png"):
            f = crop_feature(p)
            if f is not None: fs.append(f)
        if fs:
            feats.append(np.mean(np.stack(fs, 0), 0))
            # Keep TID as string to match JSON/CSV conventions reliably
            tids.append(tid_dir.name)
    
    if not feats: return {}
    X = np.stack(feats, 0)

    used_prototypes = (j_home is not None and j_away is not None)
    if used_prototypes:
        ph = crop_feature(j_home)
        pa = crop_feature(j_away)
        if ph is None or pa is None: return {}
        dh = np.linalg.norm(X - ph, axis=1)
        da = np.linalg.norm(X - pa, axis=1)
        y = (da < dh).astype(int)
    else:
        if algo == "kmeans":
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            y = km.fit_predict(X)
        elif algo == "gmm":
            gm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
            y = gm.fit_predict(X)
        else:
            return {}

    mapping = {0: "home", 1: "away"}
    return {tids[i]: mapping[y[i]] for i in range(len(tids))}

# ----------------- Half & Refine -----------------
def _in_bounds_ratio(xys):
    if not xys: return 0.0
    ok = 0
    for x, y in xys:
        if X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX: ok += 1
    return ok / len(xys)

def _score_half(invH, samples):
    pts = []
    for (cx, cy) in samples:
        v = invH @ np.array([cx, cy, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9: continue
        pts.append((float(v[0]/v[2]), float(v[1]/v[2])))
    if not pts: return 0.0, 0.0, []
    xs = [p[0] for p in pts]
    medx = float(np.median(xs))
    score = _in_bounds_ratio(pts) + 0.05 * (abs(medx) / 47.0)
    return score, medx, pts

def choose_half_auto(inv_left, inv_right, players_by_f, ball_by_f, frames, player_cls, ball_cls):
    samples = []
    step = max(1, len(frames)//25) if frames else 1
    for f in frames[::step]:
        bs = [b for b in ball_by_f.get(f, []) if b[1] == ball_cls]
        if bs:
            b = max(bs, key=lambda z: z[-1])
            samples.append((b[2] + b[4]/2.0, b[3] + b[5]/2.0))
        ps = [p for p in players_by_f.get(f, []) if p[1] == player_cls]
        for p in sorted(ps, key=lambda z: -z[-1])[:3]:
            samples.append((p[2] + p[4]/2.0, p[3] + p[5]/2.0))

    sL = medxL = sR = medxR = 0.0
    if inv_left  is not None: sL, medxL, _ = _score_half(inv_left,  samples)
    if inv_right is not None: sR, medxR, _ = _score_half(inv_right, samples)
    
    if sR > sL or (abs(sR - sL) < 1e-6 and medxR > abs(medxL)):
        return "right", (inv_right if inv_right is not None else inv_left), (sL, sR)
    return "left", (inv_left if inv_left is not None else inv_right), (sL, sR)

def apply_sim(X, Y, s=1.0, th=0.0, tx=0.0, ty=0.0):
    c, si = np.cos(th), np.sin(th)
    return s*(c*X - si*Y) + tx, s*(si*X + c*Y) + ty

def refine_similarity(invH, tracks):
    # Coarse grid search to keep projected player feet inside court bounds.
    pts = []
    for lst in tracks.values():
        for (tid, cls, x,y,w,h, conf) in lst:
            if cls != 0: continue
            cx, cy = x + w/2.0, y + h
            v = invH @ np.array([cx, cy, 1.0], float)
            if abs(v[2]) < 1e-9: continue
            pts.append((v[0]/v[2], v[1]/v[2]))
    if not pts: return (1.0, 0.0, 0.0, 0.0)
    
    best = (-1, (1.0, 0.0, 0.0, 0.0))
    for s in np.linspace(0.8, 1.2, 40):
        for th in np.deg2rad(np.linspace(-5, 5, 11)):
            for tx in np.linspace(-5, 5, 15):
                for ty in np.linspace(-5, 5, 15):
                    ok = 0
                    for X,Y in pts:
                        x2, y2 = apply_sim(X, Y, s, th, tx, ty)
                        if X_MIN <= x2 <= X_MAX and Y_MIN <= y2 <= Y_MAX: ok += 1
                    if ok > best[0]: best = (ok, (s, th, tx, ty))
    return best[1]

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val", "val10"], default="val")
    ap.add_argument("--tracker", choices=["bytetrack","botsort"], required=True)
    ap.add_argument("--game_id", required=True)
    ap.add_argument("--half", default="auto")
    ap.add_argument("--player_cls", type=int, default=0)
    ap.add_argument("--ball_cls",   type=int, default=1)
    ap.add_argument("--mot_root", default="data/meta/mot")
    ap.add_argument("--res_root", default="results/track")
    ap.add_argument("--hom_root", default="data/meta/homography")
    ap.add_argument("--out_root", default="results/court_tracks")
    ap.add_argument("--refine", default="auto")
    
    # Team Args
    ap.add_argument("--team_algo", choices=["kmeans", "gmm"], default=None)
    ap.add_argument("--crops_root", default="data/meta/team_crops")
    ap.add_argument("--jersey_home", default=None)
    ap.add_argument("--jersey_away", default=None)
    ap.add_argument("--save_team_json", default="data/meta/team_assign", 
                    help="Folder to save the resulting team json map")

    args = ap.parse_args()

    # 1. Compute & Save Team Map
    tid_to_team = {}
    if args.team_algo or (args.jersey_home and args.jersey_away):
        print(f"[INFO] Classifying teams for {args.game_id}...")
        tid_to_team = compute_team_map(args.game_id, args.crops_root, 
                                       args.team_algo, args.jersey_home, args.jersey_away)
        print(f"[INFO] Assigned teams for {len(tid_to_team)} IDs.")
        
        # --- FIX: Save to JSON so quick_plot_tracks can use it later ---
        if tid_to_team:
            out_json = Path(args.save_team_json) / f"{args.game_id}.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(tid_to_team, indent=2))
            print(f"[OK] Saved team map to {out_json}")

    half2inv = load_homographies(Path(args.hom_root), args.game_id)
    inv_left, inv_right = half2inv.get("left"), half2inv.get("right")
    
    mot_split = Path(args.mot_root)/args.split
    res_split = Path(args.res_root)/args.tracker/args.split
    out_split = Path(args.out_root)/args.split
    out_split.mkdir(parents=True, exist_ok=True)

    seq_dirs = [d for d in sorted(mot_split.iterdir()) 
                if d.is_dir() and d.name.startswith(f"{args.game_id}_")]

    for seq_dir in seq_dirs:
        seq = seq_dir.name
        mot_file = res_split/seq/f"{seq}.txt"
        if not mot_file.exists(): continue

        tracks = load_mot(mot_file)
        if not tracks: continue

        frames = sorted(tracks.keys())
        players = {f:[t for t in tracks[f] if t[1]==args.player_cls] for f in frames}
        balls   = {f:[t for t in tracks[f] if t[1]==args.ball_cls]   for f in frames}

        if args.half in ("left","right"):
            chosen = args.half
            invH = inv_left if chosen == "left" else inv_right
        else:
            chosen, invH, _ = choose_half_auto(inv_left, inv_right, players, balls, frames, args.player_cls, args.ball_cls)

        s, th, tx, ty = (1.0, 0.0, 0.0, 0.0)
        if args.refine == "auto":
            s, th, tx, ty = refine_similarity(invH, tracks)

        # WRITE CSV (Force 'team' column)
        out_csv = out_split/f"{seq}.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame","tid","cls","x_ft","y_ft","conf","half","team"])
            
            for fr, lst in sorted(tracks.items()):
                for (tid, cls, x,y,wid,hei, conf) in lst:
                    cx, cy = x + wid/2.0, (y+hei) if cls==args.player_cls else (y+hei/2.0)
                    v = invH @ np.array([cx, cy, 1.0], float)
                    if abs(v[2]) < 1e-9: continue
                    X, Y = v[0]/v[2], v[1]/v[2]
                    
                    if args.refine == "auto":
                        X, Y = apply_sim(X, Y, s, th, tx, ty)
                    
                    # Map TID (int) to Team (via string lookup)
                    team_lbl = tid_to_team.get(str(tid), "") if cls==args.player_cls else ""
                    
                    w.writerow([fr, tid, cls, f"{X:.6f}", f"{Y:.6f}", f"{conf:.4f}", chosen, team_lbl])
                    
        arr = np.loadtxt(out_csv, delimiter=",", skiprows=1, usecols=(3,4), dtype=float)
        if arr.size:
            arr = arr.reshape(-1,2)
            oob = np.sum((arr[:,0]<X_MIN)|(arr[:,0]>X_MAX)|(arr[:,1]<Y_MIN)|(arr[:,1]>Y_MAX))
            tot = arr.shape[0]
            print(f"[OK] {seq}: wrote {out_csv}  | OOB {oob}/{tot} ({(oob/tot if tot else 0):.1%})")
        else:
            print(f"[OK] {seq}: wrote {out_csv}")

if __name__ == "__main__":
    main()
