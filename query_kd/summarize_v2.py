#!/usr/bin/env python3
"""Table of all multi-student runs (results.json of run_multi.py / run_multi_v2.py)."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()
rows = ["| run | labels | students | teacher queries | single students (step C) | ensemble | final single | "
        "PGA kept / teacher agrees | classes <100 imgs | top-10 share |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for d in a.runs:
    p = os.path.join(d, "results.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    args = json.load(open(os.path.join(d, "args.json")))
    cs = [v["acc"] for v in r.get("C_students", {}).values()]
    c = f"{min(cs):.2f}-{max(cs):.2f}%" if len(cs) > 1 else (f"{cs[0]:.2f}%" if cs else "-")
    ens = f"{r['C_ensemble']['acc']:.2f}%" if "C_ensemble" in r and len(cs) > 1 else "-"
    fin = f"**{r['final_single_student']['acc']:.2f}%**" if "final_single_student" in r else "running"
    pga = r.get("pga", {})
    pg = f"{pga['kept']}/{pga['generated']} / {pga['diag_teacher_agrees_with_kept']:.1f}%" if pga else "-"
    D = r.get("D", {})
    rows.append(f"| {os.path.basename(d.rstrip('/'))} | {args['label_mode']} | {args['n_students']} | "
                f"{D.get('teacher_queries', args['total_budget']):,} | {c} | {ens} | {fin} | {pg} | "
                f"{D.get('classes_below_100', '-')} | {D.get('top10_share', 0):.1f}% |")
txt = f"Teacher accuracy: {r.get('teacher_acc', 0):.2f}%\n\n" + "\n".join(rows) if len(rows) > 2 else "no results yet"
print(txt)
if a.out:
    open(a.out, "w").write(txt + "\n")
