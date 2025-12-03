
#!/usr/bin/env python3
# Build a MOT mirror from an ordered frame list (train.txt / val.txt).
# Robust to: bare filenames, leading '/', stray directory lines, mixed extensions.
import argparse, csv, os, re, sys
from pathlib import Path
from PIL import Image

# Accept clip names like v1_p4_00001.png (extract "v1_p4")
CLIP_NAME = re.compile(r'(v\d+_p\d+)_\d+\.(jpg|jpeg|png)$', re.IGNORECASE)
IMG_EXTS = {'.jpg', '.jpeg', '.png'}

def is_image_line(s: str) -> bool:
    s = s.strip()
    if not s or s.startswith('#'):
        return False
    return Path(s).suffix.lower() in IMG_EXTS

def resolve_image(line: str, split: str, images_root: Path) -> Path:
    """
    Resolve a line from train/val list to an existing image path.
    Tries multiple candidates:
      - as-is
      - strip leading '/'
      - join with base split dir
      - join basename with base split dir
    """
    s = line.strip()
    cand = [Path(s)]
    if s.startswith('/'):
        cand.append(Path(s.lstrip('/')))
    # Map split -> actual subdir on disk
    subdir = 'train' if split == 'train' else 'val_fixed'
    base = images_root / subdir
    cand.append(base / s)
    cand.append(base / Path(s).name)

    # Common accidental prefixes we strip
    for prefix in ('images/train/', 'images/val_fixed/',
                   'data/images/train/', 'data/images/val_fixed/',
                   'data/dataset/images/train/', 'data/dataset/images/val_fixed/'):
        if s.startswith(prefix):
            tail = s[len(prefix):]
            cand.append(base / tail)

    for c in cand:
        if c.exists() and c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Could not resolve image from line: {line!r} "
                            f"searched: {[str(c) for c in cand]}")

def read_list(list_path: Path, split: str, images_root: Path):
    """Returns ordered list of (clip, img_path, ext)."""
    frames, missing = [], []
    for raw in list_path.read_text().splitlines():
        if not is_image_line(raw):
            continue
        try:
            p = resolve_image(raw, split, images_root)
            m = CLIP_NAME.search(p.name)
            if not m:
                raise ValueError(f"Bad filename (expect vX_pY_######.ext): {p.name}")
            clip = m.group(1)  # v1_p4
            ext  = p.suffix.lower()
            frames.append((clip, p, ext))
        except Exception as e:
            missing.append(f"{raw}  <-- {e}")
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} lines could not be resolved. First few:\n{sample}\n\n"
            f"HINT: use --images_root to point to the folder that contains 'train' and 'val_fixed', "
            f"e.g., --images_root Untitled/data/dataset/images"
        )
    return frames

def write_seq(seq_dir: Path, frames, fps: int):
    (seq_dir/"img1").mkdir(parents=True, exist_ok=True)
    ext = frames[0][2]  # preserve original extension
    W, H = Image.open(frames[0][1]).size
    for i, (_, src, _ext) in enumerate(frames, start=1):
        dst = seq_dir/"img1"/f"{i:06d}{ext}"
        if not dst.exists():
            try:
                os.symlink(os.path.relpath(src, dst.parent), dst)
            except Exception:
                try:
                    os.link(src, dst)
                except Exception:
                    from shutil import copy2
                    copy2(src, dst)
    (seq_dir/"seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={seq_dir.name}\n"
        "imDir=img1\n"
        f"frameRate={fps}\n"
        f"seqLength={len(frames)}\n"
        f"imWidth={W}\n"
        f"imHeight={H}\n"
        f"imExt={ext}\n"
    )
    (seq_dir/"labels.txt").write_text("pedestrian\n")  # TrackEval uses pedestrian as class name
    (seq_dir/"gt").mkdir(exist_ok=True)
    (seq_dir/"gt"/"gt.txt").write_text("")

def _float(s): return float(s.strip())
def _int(s):   return int(float(s.strip()))

def parse_master_row(row):
    """
    Auto-detect master format and return (id, x, y, w, h, conf, cls, vis) as python types.
    Supported:
      A) frame, id, x, y, w, h, conf, cls, vis
      B) frame, x, y, w, h, id, conf, cls, vis
    """
    # Heuristic keeps ingestion robust to reordered CVAT/MOT exports without manual cleanup.
    if len(row) < 6:
        raise ValueError(f"Row too short: {row}")

    # Heuristic: if col 2 looks like a float (has '.') or is very large -> it's x, so id is at col 6.
    col2 = row[1].strip()
    looks_like_float = '.' in col2 or 'e' in col2.lower()
    if looks_like_float:
        # B) frame, x, y, w, h, id, conf, cls, vis
        x, y, w, h = map(_float, row[1:5])
        tid        = _int(row[5])
        conf       = _float(row[6]) if len(row) > 6 else 1.0
        cls        = _int(row[7])   if len(row) > 7 else 1
        vis        = _float(row[8]) if len(row) > 8 else 1.0
        fmt = 'B'
    else:
        # Try A) frame, id, x, y, w, h, conf, cls, vis
        try:
            tid = _int(row[1])
            x, y, w, h = map(_float, row[2:6])
            conf       = _float(row[6]) if len(row) > 6 else 1.0
            cls        = _int(row[7])   if len(row) > 7 else 1
            vis        = _float(row[8]) if len(row) > 8 else 1.0
            fmt = 'A'
        except Exception:
            # fallback to B
            x, y, w, h = map(_float, row[1:5])
            tid        = _int(row[5])
            conf       = _float(row[6]) if len(row) > 6 else 1.0
            cls        = _int(row[7])   if len(row) > 7 else 1
            vis        = _float(row[8]) if len(row) > 8 else 1.0
            fmt = 'B*'
    return tid, x, y, w, h, conf, cls, vis, fmt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train","val"],
                    help="Use 'val' for your val_fixed set; script maps it internally.")
    ap.add_argument("--frame_list", required=True, help="data/meta/splits/train.txt or val.txt")
    ap.add_argument("--gt_master",  required=True, help="data/meta/cvat_exports/train_gt.txt or val_gt.txt")
    ap.add_argument("--out_root", default="data/meta/mot")
    ap.add_argument("--images_root", default="data/dataset/images",
                    help="Folder that contains 'train' and 'val_fixed'")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    images_root = Path(args.images_root)
    list_path   = Path(args.frame_list)
    gt_path     = Path(args.gt_master)

    assert images_root.exists(), f"--images_root not found: {images_root}"
    assert list_path.exists(),   f"--frame_list not found: {list_path}"
    assert gt_path.exists(),     f"--gt_master not found: {gt_path}"

    # 1) Read ordered frames robustly
    frames = read_list(list_path, args.split, images_root)

    # 2) Group by clip in order of first appearance
    by_clip, order, counts = {}, [], {}
    for clip, p, ext in frames:
        if clip not in by_clip:
            by_clip[clip] = []
            order.append(clip)
            counts[clip] = 0
        by_clip[clip].append((clip, p, ext))

    # 3) Create MOT seqs and global map (clip, local_frame)
    out_split = Path(args.out_root)/args.split  # we keep 'val' (not 'val_fixed') for MOT
    out_split.mkdir(parents=True, exist_ok=True)

    global_map = []
    for clip in order:
        seq_dir = out_split/clip
        write_seq(seq_dir, by_clip[clip], args.fps)
        for _ in by_clip[clip]:
            counts[clip] += 1
            global_map.append((clip, counts[clip]))

    # 4) Split master GT into per-clip gt.txt, renumber frames to local 1..N
    # Build a map: global_frame (1-based) -> (seq_name, local_frame)
    frame_to_seq = {i+1: pair for i, pair in enumerate(global_map)}  # global_map order is 1..N images

    # Read CVAT-exported MOT GT (frame,id,x,y,w,h,conf,class,vis)
    import csv
    from collections import defaultdict

    # group rows by global frame index
    by_frame = defaultdict(list)
    with open(gt_path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 6:
                continue
            gframe = int(float(row[0]))  # 1..116 in your file
            by_frame[gframe].append(row)

    # write all rows for each frame to the correct sequence, remapping frame index
    written = 0
    for gframe in sorted(by_frame.keys()):
        if gframe not in frame_to_seq:
            # labeled a frame that is not in your frame list -> skip safely
            continue
        seq, local_fr = frame_to_seq[gframe]
        gt_file = (out_split / seq / "gt" / "gt.txt")
        with gt_file.open("a", newline="") as w:
            wr = csv.writer(w)
            for row in by_frame[gframe]:
                row = list(row)
                row[0] = str(local_fr)          # put local frame number (1..N of that seq)
                wr.writerow(row)
                written += 1

    print(f"[OK] wrote {written} GT rows across {len(by_frame)} labeled frames into {out_split}")


if __name__ == "__main__":
    main()
