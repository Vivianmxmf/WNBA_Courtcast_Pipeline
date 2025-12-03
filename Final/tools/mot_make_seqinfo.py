# tools/mot_make_seqinfo.py
import argparse, glob
from pathlib import Path
import cv2, configparser

def write_seqinfo(seq_dir: Path, fps: int):
    img_dir = seq_dir / "img1"
    imgs = sorted(glob.glob(str(img_dir / "*.jpg"))) or sorted(glob.glob(str(img_dir / "*.png")))
    if not imgs:
        print(f"[skip] no images: {img_dir}")
        return
    im0 = cv2.imread(imgs[0])
    if im0 is None:
        print(f"[skip] cannot read first image: {imgs[0]}")
        return
    H, W = im0.shape[:2]
    N = len(imgs)

    cfg = configparser.ConfigParser()
    cfg["Sequence"] = {
        "name": seq_dir.name,
        "imDir": "img1",
        "frameRate": str(fps),
        "seqLength": str(N),
        "imWidth": str(W),
        "imHeight": str(H),
    }
    out = seq_dir / "seqinfo.ini"
    with open(out, "w") as f:
        cfg.write(f)
    print(f"[OK] wrote {out} (frames={N}, {W}x{H}, fps={fps})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., data/meta/mot/val10")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.root)
    for seq_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        write_seqinfo(seq_dir, args.fps)

if __name__ == "__main__":
    main()

#python3 tools/mot_make_seqinfo.py --root data/meta/mot/val10 --fps 10