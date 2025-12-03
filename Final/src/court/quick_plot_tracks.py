#!/usr/bin/env python3
import argparse, csv, json, shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Court extents
X_MIN, X_MAX = -47.0, 47.0
Y_MIN, Y_MAX = -25.0, 25.0

def _pick(keys, candidates, default=None):
    clean_keys = {k.strip(): k for k in keys}
    for c in candidates:
        if c in clean_keys:
            return clean_keys[c]
    return default

def normalize_tid_str(s):
    """Normalize '1.0' -> '1', ' 1 ' -> '1'."""
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except:
        return s

def load_team_map(team_json: Path):
    if not team_json or not team_json.exists():
        return {}
    try:
        obj = json.loads(team_json.read_text())
        # Return as-is; keys might be '1' or 'v1_p4_1'
        return {str(k).strip(): v for k, v in obj.items()}
    except Exception:
        pass
    return {}

def lookup_team(tid, team_map, seq_prefix):
    """Try to find the TID in the map using various key formats."""
    if not tid or not team_map: return None
    
    norm_id = normalize_tid_str(tid)
    
    # 1. Try exact match (e.g. key is "1")
    if norm_id in team_map:
        return team_map[norm_id]
    
    # 2. Try prefix match (e.g. key is "v1_p4_1")
    prefixed_id = f"{seq_prefix}_{norm_id}"
    if prefixed_id in team_map:
        return team_map[prefixed_id]
        
    return None

def update_csv_with_teams(csv_path: Path, team_map: dict):
    if not team_map: return

    seq_prefix = csv_path.stem # e.g. "v1_p4"
    temp_file = NamedTemporaryFile(mode='w', delete=False, newline='')
    updated_count = 0
    
    try:
        with csv_path.open('r', newline='', encoding='utf-8-sig') as infile, temp_file as outfile:
            reader = csv.DictReader(infile)
            fieldnames = list(reader.fieldnames)
            
            if "team" not in fieldnames:
                fieldnames.append("team")
            
            tid_col = _pick(fieldnames, ["tid", "id", "track_id"])
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                if tid_col:
                    raw_tid = row.get(tid_col, "")
                    found_team = lookup_team(raw_tid, team_map, seq_prefix)
                    
                    if found_team:
                        row["team"] = found_team
                        updated_count += 1
                writer.writerow(row)
                
        shutil.move(temp_file.name, csv_path)
        print(f"[INFO] Updated {csv_path}: assigned teams to {updated_count} rows.")
        
    except Exception as e:
        Path(temp_file.name).unlink(missing_ok=True)
        print(f"[WARN] Failed to update CSV: {e}")

def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        fn = [c.strip() for c in r.fieldnames] if r.fieldnames else []
        xk = _pick(fn, ["x", "x_ft"])
        yk = _pick(fn, ["y", "y_ft"])
        idk = _pick(fn, ["tid", "id"])
        clk = _pick(fn, ["cls", "class", "category"])
        ibk = _pick(fn, ["in_bounds", "inb"])
        tmk = _pick(fn, ["team", "label", "side"]) 

        if not xk or not yk:
            print(f"[ERR] Columns found: {fn}")
            raise SystemExit(f"[err] {csv_path}: cannot find x/y columns.")

        for row in r:
            try:
                x = float(row[xk]); y = float(row[yk])
            except: continue
            
            if ibk and row.get(ibk, "1").lower() not in ("1", "true"):
                continue

            tid_str = row.get(idk, "")
            cls = row.get(clk)
            try: cls = int(float(cls)) if cls else None
            except: cls = None
            
            team = row.get(tmk, "").strip()
            
            rows.append({"x": x, "y": y, "tid_str": tid_str, "cls": cls, "team": team})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--team_json", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--max_points", type=int, default=20000)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    team_map = load_team_map(Path(args.team_json)) if args.team_json else {}

    # Debug info
    if team_map:
        keys = list(team_map.keys())
        print(f"[INFO] Loaded map with {len(keys)} keys. Sample: {keys[:3]}")

    if team_map and csv_path.exists():
        update_csv_with_teams(csv_path, team_map)

    rows = load_rows(csv_path)
    if not rows: raise SystemExit(f"[err] no rows in {csv_path}")

    # Plotting
    plt.style.use("default")
    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(111, facecolor="#0f1116")
    ax.set_xlim(X_MIN, X_MAX); ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal")
    ax.grid(True, color="#2a2f3a", linewidth=0.6, alpha=0.6)
    ax.add_patch(plt.Rectangle((X_MIN, Y_MIN), X_MAX-X_MIN, Y_MAX-Y_MIN, fill=False, edgecolor="#6b7280", lw=1.2))

    xs_home, ys_home = [], []
    xs_away, ys_away = [], []
    xs_ball, ys_ball = [], []
    xs_other, ys_other = [], []

    seq_prefix = csv_path.stem
    
    for i, r in enumerate(rows):
        if i >= args.max_points: break
        
        lbl = r["team"]
        # Fallback lookup if CSV column was empty but map exists
        if not lbl and team_map:
            lbl = lookup_team(r["tid_str"], team_map, seq_prefix)

        if r["cls"] == 1:
            xs_ball.append(r["x"]); ys_ball.append(r["y"])
        elif lbl == "home":
            xs_home.append(r["x"]); ys_home.append(r["y"])
        elif lbl == "away":
            xs_away.append(r["x"]); ys_away.append(r["y"])
        else:
            xs_other.append(r["x"]); ys_other.append(r["y"])

    if xs_home:  ax.scatter(xs_home, ys_home, s=14, c="#00A2FF", alpha=0.9, ec="#E5F2FF", lw=0.3, label="home")
    if xs_away:  ax.scatter(xs_away, ys_away, s=14, c="#FF3D8B", alpha=0.9, ec="#FFD6E8", lw=0.3, label="away")
    if xs_other: ax.scatter(xs_other, ys_other, s=10, c="#555555", alpha=0.6, ec="none", label="other")
    if xs_ball:  ax.scatter(xs_ball, ys_ball, s=26, c="#FFD84D", alpha=1.0, ec="#2B2B2B", lw=0.6, label="ball")

    ax.set_xlabel("X (ft)", color="#d1d5db"); ax.set_ylabel("Y (ft)", color="#d1d5db")
    ax.tick_params(colors="#9ca3af")
    if args.title: ax.set_title(args.title, color="#e5e7eb")
    if any([xs_home, xs_away, xs_other, xs_ball]):
        leg = ax.legend(facecolor="#111319", edgecolor="#2a2f3a", labelcolor="#e5e7eb")
        for t in leg.get_texts(): t.set_color("#e5e7eb")

    fig.tight_layout()
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] wrote {args.out} ({min(len(rows), args.max_points)} points)")

if __name__ == "__main__":
    main()