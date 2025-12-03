#!/usr/bin/env python3
import argparse, csv, re, json
from pathlib import Path

try:
    import yaml  # used to read args.yaml if available
except Exception:
    yaml = None

# -------- helpers --------
SEQ_RE = re.compile(r'(v\d+_p\d+)')

def imsize_from_seqinfo(p: Path):
    W = H = None
    for line in p.read_text().splitlines():
        if line.startswith("imWidth="):  W = int(line.split("=",1)[1])
        if line.startswith("imHeight="): H = int(line.split("=",1)[1])
    if W is None or H is None:
        raise RuntimeError(f"Missing imWidth/imHeight in {p}")
    return W, H

def seq_len_from_seqinfo(p: Path):
    for line in p.read_text().splitlines():
        if line.startswith("seqLength="):
            return int(line.split("=",1)[1])
    return None

def seq_from_args_yaml(run_dir: Path):
    for fn in ("args.yaml","args.json","cfg.yaml"):
        f = run_dir / fn
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text()) if f.suffix==".json" else (yaml.safe_load(f.read_text()) if yaml else None)
            if not data:
                continue
            src = str(data.get("source",""))
            # expect .../data/meta/mot/<split>/<seq>/img1
            m = re.search(r"/(v\d+_p\d+)/img1/?$", src)
            if m:
                return m.group(1)
        except Exception:
            pass
    return None

def seq_from_run_name(name: str):
    m = SEQ_RE.search(name)
    return m.group(1) if m else None

def find_labels_dir(run_dir: Path, seq: str|None):
    # preferred layouts
    cand = []
    if (run_dir/"labels").exists():
        cand.append(run_dir/"labels")
    if seq and (run_dir/seq/"labels").exists():
        cand.append(run_dir/seq/"labels")
    # fallback: any labels dir with .txt
    cand.extend([d for d in run_dir.rglob("labels")])
    for lab in cand:
        txts = sorted(lab.glob("*.txt"))
        if txts:
            return lab, txts
    return None, []

def parse_ultra_line(line: str):
    # "cls xc yc w h conf id"
    t = re.split(r"\s+", line.strip())
    if len(t) < 7: return None
    try:
        cls = int(float(t[0]))
        xc, yc, ww, hh = map(float, t[1:5])
        conf = float(t[5])
        tid  = int(float(t[6]))
        return cls, xc, yc, ww, hh, conf, tid
    except Exception:
        return None

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, choices=["botsort","bytetrack"])
    ap.add_argument("--split", required=True, choices=["train","val"])
    ap.add_argument("--ultra_root", default="results/track",
                    help="root where YOLO runs were saved")
    ap.add_argument("--mot_root", default="data/meta/mot",
                    help="MOT mirror (for seq list + sizes)")
    ap.add_argument("--out_root", default="results/track",
                    help="where to write MOT txt files")
    args = ap.parse_args()

    # collect run dirs like botsort_val_v1_p4, botsort_val_v3_p6, ...
    pattern = f"{args.tracker}_{args.split}"
    ultra_root = Path(args.ultra_root)
    run_dirs = sorted([d for d in ultra_root.iterdir()
                       if d.is_dir() and d.name.startswith(pattern)])
    if not run_dirs:
        raise SystemExit(f"No run dirs matching {pattern}* under {ultra_root}")

    mot_split = Path(args.mot_root)/args.split
    seq_dirs = {d.name: d for d in mot_split.iterdir() if d.is_dir()}
    if not seq_dirs:
        raise SystemExit(f"No MOT sequences under {mot_split}")

    out_split = Path(args.out_root)/args.tracker/args.split
    out_split.mkdir(parents=True, exist_ok=True)

    done = set()
    for rd in run_dirs:
        # Determine seq: args.yaml first, otherwise from folder name
        seq = seq_from_args_yaml(rd) or seq_from_run_name(rd.name)
        if not seq or seq not in seq_dirs:
            # last resort: try match by frame count
            lab_dir, frames = find_labels_dir(rd, None)
            if not frames:
                print(f"[WARN] {rd.name}: labels not found")
                continue
            fcount = len(frames)
            candidates = [s for s, sdir in seq_dirs.items()
                          if seq_len_from_seqinfo(sdir/'seqinfo.ini') == fcount]
            seq = candidates[0] if len(candidates)==1 else None
            if not seq:
                print(f"[INFO] {rd.name}: could not map to a unique seq (frames={fcount})")
                continue

        # locate labels and convert
        lab_dir, frame_files = find_labels_dir(rd, seq)
        if not frame_files:
            print(f"[WARN] {rd.name}: empty labels dir for seq {seq}")
            continue

        W, H = imsize_from_seqinfo(seq_dirs[seq]/"seqinfo.ini")
        out_dir = out_split/seq
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir/f"{seq}.txt"
        with out_file.open("w", newline="") as f:
            wcsv = csv.writer(f)
            for ff in sorted(frame_files):
                m = re.search(r"(\d+)$", ff.stem)
                if not m:  # skip weird names
                    continue
                fr = int(m.group(1))
                for line in ff.read_text().splitlines():
                    parsed = parse_ultra_line(line)
                    if not parsed:
                        continue
                    cls, xc, yc, ww, hh, conf, tid = parsed
                    x = (xc - ww/2.0) * W
                    y = (yc - hh/2.0) * H
                    w = ww * W
                    h = hh * H
                    mot_cls = cls + 1  # 0->1 (player), 1->2 (ball)
                    wcsv.writerow([fr, tid, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}",
                                   f"{conf:.4f}", mot_cls, 1])

        print(f"[OK] {seq}: wrote {out_file}")
        done.add(seq)

    missing = sorted(set(seq_dirs.keys()) - done)
    if missing:
        print(f"\n[INFO] No converted results for {len(missing)} seqs: {', '.join(missing[:12])}"
              f"{' ...' if len(missing)>12 else ''}")
        print("      Tip: ensure run folder names contain the seq (e.g., botsort_val_v1_p4)")
        print("           and that 'labels/*.txt' exists inside each run dir.")

if __name__ == "__main__":
    main()
