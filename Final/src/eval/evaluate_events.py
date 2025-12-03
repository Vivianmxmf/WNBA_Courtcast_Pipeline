#!/usr/bin/env python3
"""
Event-level evaluator with frame tolerance and dribble IoU.
- Gold format (CSV per sequence in data/meta/actions_gold/<split>/):
    Required columns:
      type ∈ {pass, shot, dribble}
      For pass : t_release, t_catch
      For shot : t_release
      For dribble: t_start, t_end
  Extra columns are ignored.

- Auto format (JSON per sequence from rules_actions.py):
    {"sequence": "...", "events":[ ... same keys as above ... ]}

Matching:
  pass   -> greedy bipartite with |Δt_release|<=tol and |Δt_catch|<=tol
  shot   -> |Δt_release|<=tol
  dribble-> IoU >= dribble_iou on [start,end] windows

Writes a single CSV summary and (optional) small per-pass visualization JSONs.

Usage:
  python3 src/eval/evaluate_events.py \
      --gold_glob "data/meta/actions_gold/val/*.csv" \
      --auto_glob "results/actions_auto/val/*.json" \
      --tol 1 --dribble_iou 0.5 \
      --out_csv results/eval/events_val_summary.csv \
      --export_viz_dir results/eval/pass_viz
"""

import argparse, csv, glob, json
from pathlib import Path


def read_gold_csv(path):
    G = {"pass":[], "shot":[], "dribble":[]}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            typ = row.get("type","").strip().lower()
            if typ not in G: continue
            if typ=="pass":
                G["pass"].append({"t_release": int(row["t_release"]),
                                  "t_catch":  int(row["t_catch"])})
            elif typ=="shot":
                G["shot"].append({"t_release": int(row["t_release"])})
            else:
                # dribble
                G["dribble"].append({"t_start": int(row["t_start"]),
                                     "t_end":   int(row["t_end"])})
    return G

def read_auto_json(path):
    A = {"pass":[], "shot":[], "dribble":[]}
    obj = json.loads(Path(path).read_text())
    for e in obj.get("events", []):
        typ = e.get("type")
        if typ not in A: continue
        A[typ].append(e)
    return A

# --------------- matching helpers ---------------
def greedy_match_pass(G, A, tol):
    used = set()
    tp = 0
    viz = []
    for g in G:
        best = None; best_cost = 10**9; best_j = -1
        for j,a in enumerate(A):
            if j in used: continue
            dr = abs(a.get("t_release", -10**9) - g["t_release"])
            dc = abs(a.get("t_catch",  -10**9) - g["t_catch"])
            if dr<=tol and dc<=tol:
                cost = dr + dc
                if cost < best_cost:
                    best_cost = cost; best = a; best_j = j
        if best is not None:
            used.add(best_j); tp += 1
            viz.append({"g": g, "a": best})
    fp = len(A) - tp
    fn = len(G) - tp
    return tp, fp, fn, viz

def greedy_match_shot(G, A, tol):
    used = set(); tp = 0
    for g in G:
        best = None; best_d = 10**9; best_j = -1
        for j,a in enumerate(A):
            if j in used: continue
            dr = abs(a.get("t_release", -10**9) - g["t_release"])
            if dr <= tol and dr < best_d:
                best = a; best_d = dr; best_j = j
        if best is not None:
            used.add(best_j); tp += 1
    fp = len(A) - tp
    fn = len(G) - tp
    return tp, fp, fn

def iou_1d(a0,a1,b0,b1):
    inter = max(0, min(a1,b1) - max(a0,b0) + 1)
    union = (a1-a0+1) + (b1-b0+1) - inter
    return inter/union if union>0 else 0.0

def greedy_match_dribble(G, A, iou_thr):
    used=set(); tp=0
    for g in G:
        best=None; best_iou=0.0; best_j=-1
        for j,a in enumerate(A):
            if j in used: continue
            iou = iou_1d(g["t_start"], g["t_end"],
                         a.get("t_start", -10**9), a.get("t_end", -10**9))
            if iou>=iou_thr and iou>best_iou:
                best=a; best_iou=iou; best_j=j
        if best is not None:
            used.add(best_j); tp += 1
    fp = len(A) - tp
    fn = len(G) - tp
    return tp, fp, fn

# --------------- main ---------------
def prf(tp, fp, fn):
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    return prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_glob", required=True)
    ap.add_argument("--auto_glob", required=True)
    ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--dribble_iou", type=float, default=0.5)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--export_viz_dir", default=None)
    args = ap.parse_args()

    # index by sequence stem
    gold = {}
    for fp in glob.glob(args.gold_glob):
        gold[Path(fp).stem] = read_gold_csv(fp)

    auto = {}
    for fp in glob.glob(args.auto_glob):
        auto[Path(fp).stem] = read_auto_json(fp)

    # evaluate only on overlap
    seqs = sorted(set(gold.keys()) & set(auto.keys()))
    if not seqs:
        raise SystemExit("[err] No overlapping sequences between gold and auto.")

    totals = {"pass":[0,0,0], "shot":[0,0,0], "dribble":[0,0,0]}
    if args.export_viz_dir:
        Path(args.export_viz_dir).mkdir(parents=True, exist_ok=True)

    for s in seqs:
        G, A = gold[s], auto[s]
        tp,fp,fn, viz = greedy_match_pass(G["pass"], A["pass"], args.tol)
        totals["pass"][0]+=tp; totals["pass"][1]+=fp; totals["pass"][2]+=fn
        if args.export_viz_dir and viz:
            Path(args.export_viz_dir, f"{s}_pass_pairs.json").write_text(json.dumps(viz, indent=2))

        tp,fp,fn = greedy_match_shot(G["shot"], A["shot"], args.tol)
        totals["shot"][0]+=tp; totals["shot"][1]+=fp; totals["shot"][2]+=fn

        tp,fp,fn = greedy_match_dribble(G["dribble"], A["dribble"], args.dribble_iou)
        totals["dribble"][0]+=tp; totals["dribble"][1]+=fp; totals["dribble"][2]+=fn

    # summary CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Type","TP","FP","FN","Prec","Rec","F1"])
        grand = [0,0,0]
        for typ in ["pass","shot","dribble"]:
            tp,fp,fn = totals[typ]
            p,r,f1 = prf(tp,fp,fn)
            w.writerow([typ, tp,fp,fn, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])
            grand[0]+=tp; grand[1]+=fp; grand[2]+=fn
        p,r,f1 = prf(*grand)
        w.writerow(["ALL", grand[0],grand[1],grand[2], f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])
    print(out.read_text())

if __name__ == "__main__":
    main()