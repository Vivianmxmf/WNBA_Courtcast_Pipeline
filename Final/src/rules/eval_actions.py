#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def load(path): 
    obj=json.loads(Path(path).read_text()); 
    ev=[e for e in obj.get("events",[]) if not e.get("ignore",False)]
    return obj.get("sequence"), obj.get("fps",30), ev

def match_pass(g,a,win=3):
    used=set(); tp=fp=fn=0
    for eg in g:
        best=-1; bi=None
        for i,ea in enumerate(a):
            if i in used: continue
            if ea["type"]!="pass": continue
            if eg.get("from_id")!=ea.get("from_id"): continue
            if eg.get("to_id")  !=ea.get("to_id")  : continue
            if abs(eg["t_release"]-ea["t_release"])<=win and abs(eg["t_catch"]-ea["t_catch"])<=win:
                d=abs(eg["t_release"]-ea["t_release"])+abs(eg["t_catch"]-ea["t_catch"])
                if best<0 or d<best: best=d; bi=i
        if bi is not None: tp+=1; used.add(bi)
        else: fn+=1
    fp += sum(1 for i,ea in enumerate(a) if ea["type"]=="pass" and i not in used)
    return tp,fp,fn

def match_shot(g,a,win=3):
    used=set(); tp=fp=fn=0
    for eg in g:
        best=-1; bi=None
        for i,ea in enumerate(a):
            if i in used: continue
            if ea["type"] not in ("shot","layup"): continue
            if eg.get("shooter_id")!=ea.get("shooter_id"): continue
            if abs(eg["t_release"]-ea["t_release"])<=win:
                d=abs(eg["t_release"]-ea["t_release"])
                if best<0 or d<best: best=d; bi=i
        if bi is not None: tp+=1; used.add(bi)
        else: fn+=1
    fp += sum(1 for i,ea in enumerate(a) if ea["type"] in ("shot","layup") and i not in used)
    return tp,fp,fn

def iou_1d(a0,a1,b0,b1):
    inter=max(0, min(a1,b1)-max(a0,b0)+1)
    union=(a1-a0+1)+(b1-b0+1)-inter
    return inter/union if union>0 else 0.0

def match_dribble(g,a,thr=0.3):
    used=set(); tp=fp=fn=0
    for eg in g:
        best=-1; bi=None
        for i,ea in enumerate(a):
            if i in used: continue
            if ea["type"]!="dribble": continue
            if eg.get("player_id")!=ea.get("player_id"): continue
            iou=iou_1d(eg["t_start"],eg["t_end"],ea["t_start"],ea["t_end"])
            if iou>=thr and iou>best: best=iou; bi=i
        if bi is not None: tp+=1; used.add(bi)
        else: fn+=1
    fp += sum(1 for i,ea in enumerate(a) if ea["type"]=="dribble" and i not in used)
    return tp,fp,fn

def prf(tp,fp,fn):
    p = tp/(tp+fp) if (tp+fp)>0 else 0.0
    r = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return p,r,f

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)   # results/actions_gold/val/v1_p4.json
    ap.add_argument("--auto", required=True)   # results/actions_auto/val/v1_p4.json
    ap.add_argument("--pass_win", type=int, default=3)
    ap.add_argument("--shot_win", type=int, default=3)
    ap.add_argument("--dribble_iou", type=float, default=0.3)
    args=ap.parse_args()

    _,_,G = load(args.gold)
    _,_,A = load(args.auto)
    Gp=[e for e in G if e["type"]=="pass"]
    Ap=[e for e in A if e["type"]=="pass"]
    Gs=[e for e in G if e["type"] in ("shot","layup")]
    As=[e for e in A if e["type"] in ("shot","layup")]
    Gd=[e for e in G if e["type"]=="dribble"]
    Ad=[e for e in A if e["type"]=="dribble"]

    tp,fp,fn = match_pass(Gp,Ap,args.pass_win); Pp,Rp,Fp = prf(tp,fp,fn)
    ts,fs,ns = match_shot(Gs,As,args.shot_win); Ps,Rs,Fs = prf(ts,fs,ns)
    td,fd,nd = match_dribble(Gd,Ad,args.dribble_iou); Pd,Rd,Fd = prf(td,fd,nd)

    print(f"PASS    TP={tp} FP={fp} FN={fn}  P={Pp:.3f} R={Rp:.3f} F1={Fp:.3f}")
    print(f"SHOT    TP={ts} FP={fs} FN={ns}  P={Ps:.3f} R={Rs:.3f} F1={Fs:.3f}")
    print(f"DRIBBLE TP={td} FP={fd} FN={nd}  P={Pd:.3f} R={Rd:.3f} F1={Fd:.3f}")

if __name__=="__main__":
    main()