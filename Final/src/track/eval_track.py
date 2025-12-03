
#!/usr/bin/env python3
import argparse, subprocess, sys, shutil
from pathlib import Path

def rebuild_tracker_dir(res_root: Path) -> Path:
    """
    Build TrackEval-friendly layout:
      <res_root>/tracker/data/<seq>.txt
    e.g., results/track/botsort/val/tracker/data/v1_p4.txt
    """
    tracker_dir = res_root / "tracker" / "data"
    if tracker_dir.parent.exists():
        shutil.rmtree(tracker_dir.parent)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    for seq_dir in sorted(res_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        src = seq_dir / f"{seq}.txt"
        if not src.exists():
            continue
        shutil.copy2(src, tracker_dir / f"{seq}.txt")
        made += 1
    if made == 0:
        raise SystemExit(f"No <seq>/<seq>.txt files found under {res_root}")
    print(f"[prep] copied {made} sequences into {tracker_dir}")
    return tracker_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_gt", required=True, help="MOT-style GT split root, e.g. data/meta/mot/val")
    ap.add_argument("--res_root", required=True, help="results/track/{botsort|bytetrack}/val")
    ap.add_argument("--classes", required=True, help="MOT class id(s), e.g. 1 (players) or 2 (ball)")
    ap.add_argument("--cores", type=int, default=8)
    args = ap.parse_args()

    gt = Path(args.mot_gt).resolve()
    res_root = Path(args.res_root).resolve()

    _ = rebuild_tracker_dir(res_root)

    py = sys.executable  # use the interpreter you're running now
    # NOTE: call the actual TrackEval script module:
    cmd = [
        py, "-m", "trackeval.scripts.run_mot_challenge",
        "--BENCHMARK", "MOT17",
        "--GT_FOLDER", str(gt),
        "--TRACKERS_FOLDER", str(res_root),
        "--TRACKERS_TO_EVAL", "tracker",
        "--METRICS", "HOTA", "CLEAR", "Identity", "Count",
        "--USE_CLASS_INFO", "True",
        "--CLASSES_TO_EVAL", str(args.classes),
        "--DO_PREPROC", "False",
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", str(args.cores),
    ]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()