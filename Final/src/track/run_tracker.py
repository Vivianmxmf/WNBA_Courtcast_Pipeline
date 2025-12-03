#!/usr/bin/env python3
import argparse, subprocess, re
from pathlib import Path

def run_seq(model, img_dir, tracker_yaml, classes, conf, iou, project, run_name):
    # Drive Ultralytics tracker CLI for a single MOT sequence and persist txt + visuals.
    args = [
        "yolo", "track",
        f"model={model}",
        f"source={img_dir}",
        f"tracker={tracker_yaml}",
        f"classes={','.join(map(str, classes))}",
        f"conf={conf}",
        f"iou={iou}",
        "save=True",
        "save_txt=True",
        f"project={project}",
        f"name={run_name}",
    ]
    cmd = " ".join(args)
    print(">>>", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","val","val10"], default="val")
    ap.add_argument("--tracker", choices=["bytetrack","botsort"], required=True)
    ap.add_argument("--model", default="runs/detect/y8x_smallobj_third/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.5)
    ap.add_argument("--classes", default="0", help="Comma list, e.g. '0' or '0,1'")
    ap.add_argument("--mot_root", default="data/meta/mot")
    ap.add_argument("--project", default="results/track")
    ap.add_argument("--name", default=None)   # optional prefix; seq is appended if not None
    ap.add_argument("--seq", default=None, help="optional single sequence (e.g., v1_p4)")
    args = ap.parse_args()

    # parse classes like "0", "0,1", "0,"
    classes = [int(x) for x in re.split(r"[,\s]+", args.classes.strip().strip(",")) if x != ""]

    split_dir = Path(args.mot_root) / args.split
    tracker_yaml = Path("configs/tracker") / f"{args.tracker}.yaml"
    if not tracker_yaml.exists():
        raise FileNotFoundError(f"Tracker YAML not found: {tracker_yaml}")

    # which sequences to process
    seq_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if args.seq:
        seq_dirs = [d for d in seq_dirs if d.name == args.seq]
        if not seq_dirs:
            raise FileNotFoundError(f"No such sequence in {split_dir}: {args.seq}")

    for seq in seq_dirs:
        img_dir = seq / "img1"
        if not img_dir.exists():
            print(f"[WARN] missing img1 for {seq.name}, skipping")
            continue
        # per-sequence run name
        run_name = (args.name + f"_{seq.name}") if args.name else f"{args.tracker}_{args.split}_{seq.name}"
        run_seq(args.model, img_dir, tracker_yaml, classes, args.conf, args.iou, args.project, run_name)

if __name__ == "__main__":
    main()
