# tools/mot_fix_frame_ids.py
import argparse, csv, glob, math
from pathlib import Path

def count_images(seq_dir: Path):
    imgs = sorted(glob.glob(str(seq_dir/"img1" / "*.jpg"))) or sorted(glob.glob(str(seq_dir/"img1" / "*.png")))
    return len(imgs)

def max_frame_in_mot(mot_txt: Path):
    mf = 0
    with open(mot_txt) as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 6: continue
            try:
                fr = int(float(row[0]))
                if fr > mf: mf = fr
            except:
                pass
    return mf

def write_rows(rows, out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)

def remap_seq(mot_txt: Path, seq_dir: Path):
    N_img = count_images(seq_dir)
    F_old = max_frame_in_mot(mot_txt)
    if N_img <= 0 or F_old <= 0:
        print(f"[skip] N_img={N_img}, F_old={F_old} for {mot_txt}")
        return
    if N_img == F_old:
        print(f"[ok] already aligned: {mot_txt} (N={N_img})")
        return

    # read all rows
    with open(mot_txt) as f:
        rows = [r for r in csv.reader(f) if r and len(r) >= 8]

    new_rows = []
    if N_img % F_old == 0:
        k = N_img // F_old
        for r in rows:
            f_old = int(float(r[0]))
            start = (f_old - 1) * k + 1
            for i in range(k):
                r_new = r.copy()
                r_new[0] = str(start + i)
                new_rows.append(r_new)
    else:
        # nearest-neighbor stretch
        for r in rows:
            f_old = int(float(r[0]))
            f_new = 1 + round((f_old - 1) * (N_img - 1) / max(1, (F_old - 1)))
            r_new = r.copy()
            r_new[0] = str(f_new)
            new_rows.append(r_new)

    # sort by new frame then id
    new_rows.sort(key=lambda x: (int(x[0]), int(float(x[1]))))
    write_rows(new_rows, mot_txt)
    print(f"[fixed] {mot_txt.name}: {F_old} -> {N_img} frames")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_dir", required=True, help="results/track/botsort/val10")
    ap.add_argument("--gt_root", required=True, help="data/meta/mot/val10")
    args = ap.parse_args()

    mot_dir = Path(args.mot_dir)
    gt_root = Path(args.gt_root)

    for seq_dir in sorted([d for d in gt_root.iterdir() if d.is_dir()]):
        seq = seq_dir.name
        mot_txt = mot_dir / seq / f"{seq}.txt"
        if not mot_txt.exists():
            print(f"[skip] no MOT: {mot_txt}")
            continue
        remap_seq(mot_txt, seq_dir)

if __name__ == "__main__":
    main()