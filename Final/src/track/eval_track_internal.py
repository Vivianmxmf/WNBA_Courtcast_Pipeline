#!/usr/bin/env python3
# --- NumPy 2.x compat (legacy aliases) ---------------------------------------
import numpy as np
for _name, _py in [('float', float), ('int', int), ('bool', bool), ('object', object), ('long', int)]:
    if not hasattr(np, _name):
        setattr(np, _name, _py)
# -----------------------------------------------------------------------------
import argparse, csv, shutil, os
from pathlib import Path
import trackeval

# ---------- helpers ----------
def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def has_class_column(gt_file: Path) -> bool:
    """Return True if any non-empty line in gt_file has >= 8 columns (i.e., class)."""
    if not gt_file.exists():
        return False
    with gt_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = [c for c in line.split(',') if c != ""]
            return len(cols) >= 8
    return False

def copy_or_filter_gt(gt_in_root: Path, gt_out_root: Path, class_id: int):
    """
    If GT lines have >=8 columns, filter by class (col 8 == class_id).
    Otherwise copy as-is (no class info).
    Also writes seqmaps/custom.txt listing the sequences.
    """
    ensure_clean_dir(gt_out_root)
    seqmap_dir = gt_out_root / "seqmaps"
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    seq_names = []

    for seq_dir in sorted(gt_in_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        src_gt = seq_dir / "gt" / "gt.txt"
        src_ini = seq_dir / "seqinfo.ini"
        if not src_gt.exists():
            continue

        seq_out = gt_out_root / seq
        (seq_out / "gt").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_ini, seq_out / "seqinfo.ini")

        cls_mode = has_class_column(src_gt)
        dst_gt = seq_out / "gt" / "gt.txt"
        kept = 0
        with src_gt.open() as fin, dst_gt.open("w", newline="") as fout:
            w = csv.writer(fout)
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                cols = [c for c in line.split(',') if c != ""]
                if cls_mode and class_id is not None and class_id >= 0:
                    # Only keep rows whose 8th column equals class_id
                    if len(cols) < 8:
                        continue
                    if int(float(cols[7])) != int(class_id):
                        continue
                    kept += 1
                    w.writerow(cols)
                else:
                    # No class filtering (either no class column or disabled via -1)
                    kept += 1
                    w.writerow(cols)
        # include seq if any lines remained
        if kept > 0:
            seq_names.append(seq)

    # write seqmap
    with (seqmap_dir / "custom.txt").open("w") as f:
        f.write("name\n")
        for s in seq_names:
            f.write(f"{s}\n")

    print(f"[prep] GT ready at {gt_out_root} (seqs: {len(seq_names)})")
    return (seqmap_dir / "custom.txt"), len(seq_names)

def rebuild_results_folder(res_root: Path, out_root: Path):
    """
    TrackEval expects: <out_root>/tracker/data/<seq>.txt
    Our results are <res_root>/<seq>/<seq>.txt.
    """
    dst = out_root / "tracker" / "data"
    ensure_clean_dir(dst.parent)
    dst.mkdir(parents=True, exist_ok=True)

    made = 0
    for seq_dir in sorted(res_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        src = seq_dir / f"{seq}.txt"
        if src.exists():
            shutil.copy2(src, dst / f"{seq}.txt")
            made += 1
    print(f"[prep] results copied: {made} seqs → {dst}")
    return made

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot_gt", required=True, help="data/meta/mot/val")
    ap.add_argument("--res_root", required=True, help="results/track/{botsort|bytetrack}/val")
    ap.add_argument("--class_id", type=int, default=-1,
                help="GT class to keep (8th col). Use -1 to disable class filtering.")

    args = ap.parse_args()

    mot_gt = Path(args.mot_gt).resolve()
    res_root = Path(args.res_root).resolve()

    evaltmp = Path("results/track/evaltmp")
    gt_out   = evaltmp / f"gt_cls{args.class_id}"
    res_out  = evaltmp / f"res_cls{args.class_id}"
    ensure_clean_dir(evaltmp)  # start fresh each run

    # 1) Prepare GT (class-aware when possible)
    seqmap_file, nseq = copy_or_filter_gt(mot_gt, gt_out, args.class_id)
    if nseq == 0:
        raise SystemExit("[err] No GT sequences after filtering. Check your mot_gt path or class_id.")

    # 2) Prepare results
    nres = rebuild_results_folder(res_root, res_out)
    if nres == 0:
        raise SystemExit("[err] No result sequences found in res_root.")

    # 3) Dataset config
    dataset_config = {
        "BENCHMARK": "MOT17",
        "GT_FOLDER": str(gt_out),
        "TRACKERS_FOLDER": str(res_out),
        "TRACKERS_TO_EVAL": ["tracker"],
        "SPLIT_TO_EVAL": "",
        "DO_PREPROC": False,
        "USE_CLASS_INFO": False,         # safe even when GT had class; we filtered already
        "CLASSES_TO_EVAL": ["pedestrian"],
        "TRACKER_SUB_FOLDER": "data",
        "SEQMAP_FILE": str(seqmap_file),
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": True,
    }

    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)

    # 4) Run metrics one-by-one to avoid duplicate-name bug
    metric_constructors = [
        trackeval.metrics.HOTA,
        trackeval.metrics.CLEAR,
        trackeval.metrics.Identity,
    ]

    for ctor in metric_constructors:
        metric = ctor()
        name = metric.get_name()
        print("\n" + "="*62)
        print(f"Running metric: {name}")

        eval_config_single = {
            "USE_PARALLEL": False,
            "NUM_PARALLEL_CORES": 1,
            "PRINT_RESULTS": True,
            "PLOT_CURVES": False,
            "FAIL_ON_ERROR": True,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "OUTPUT_EMPTY_CLASSES": True,
            "TIME_PROGRESS": True,
            "BREAK_ON_ERROR": True,
            "RETURN_ON_ERROR": False,
            "DISPLAY_LESS_PROGRESS": True,
        }
        evaluator = trackeval.Evaluator(eval_config_single)
        evaluator.evaluate([dataset], [metric])

if __name__ == "__main__":
    main()