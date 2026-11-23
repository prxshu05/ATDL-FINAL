#!/usr/bin/env python3
"""All figures + the numbers table (results.json) for the final report. Reads only saved experiment outputs."""
import json
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from query_kd.models import build_model, load_weights  # noqa: E402
from query_kd.oracle import predict_logprobs  # noqa: E402

E2, E3 = f"{ROOT}/experiments/v2_kd", f"{ROOT}/experiments/v3_kd"
FIG = f"{ROOT}/report/figs"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
                     "font.family": "STIXGeneral", "mathtext.fontset": "stix", "axes.titleweight": "normal",
                     "axes.titlesize": 10.5, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, path):
    fig.savefig(path, dpi=220)
    fig.savefig(path.replace(".png", ".pdf"))
C_V2, C_BASE, C_PROF, C_LIT = "#1f5fa8", "#9a9a9a", "#d98c1f", "#b03a48"
NAMES = pickle.load(open(f"{ROOT}/cifar-100-python/meta", "rb"), encoding="latin1")["fine_label_names"]


def R(name):
    for d in (E3, E2):
        p = f"{d}/{name}/results.json"
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(name)


def fin(name):
    return R(name)["final_single_student"]["acc"]


def ens(name):
    r = R(name)
    return r.get("C_ensemble", r["final_single_student"])["acc"]


def ms(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------------------------------------------------------------- numbers used in the report
num = {}
for L in ("soft", "hard"):
    num[L] = {
        "v2_final": ms([fin(f"v2_{L}"), fin(f"seed2_{L}"), fin(f"seed3_{L}")]),
        "v2_ens": ms([ens(f"v2_{L}"), ens(f"seed2_{L}"), ens(f"seed3_{L}")]),
        "base": ms([fin(f"base_{L}"), fin(f"base_seed3_{L}")]),
        "prof": fin(f"prof_{L}"),
        "b50k": fin(f"b50k_{L}"), "b100k": fin(f"b100k_{L}"), "b500k": fin(f"b500k_{L}"),
        "b500k_ens": ens(f"b500k_{L}"), "base500k": fin(f"base500k_{L}"),
        "b50k_ens": ens(f"b50k_{L}"), "b100k_ens": ens(f"b100k_{L}"),
        "v3": fin(f"v3_{L}"), "v3b100k": fin(f"v3b100k_{L}"),
        "nobal": fin(f"nobal_{L}"), "one": fin(f"one_{L}"), "nodiv": fin(f"nodiv_{L}"),
        "nopga": fin(f"nopga_{L}"), "lam1": fin(f"lam1_{L}"), "mix": fin(f"mix_{L}"), "sim": fin(f"sim_{L}"),
        "v2s1": fin(f"v2_{L}"), "v2s1_ens": ens(f"v2_{L}"),
        "C_range": [min(v["acc"] for v in R(f"v2_{L}")["C_students"].values()),
                    max(v["acc"] for v in R(f"v2_{L}")["C_students"].values())],
        "pga": R(f"v2_{L}")["pga"], "pga_nodiv": R(f"nodiv_{L}")["pga"], "pga_lam1": R(f"lam1_{L}")["pga"],
        "rounds": R(f"v2_{L}")["selection_rounds"], "A_ens": R(f"v2_{L}")["A_ensemble"]["acc"],
        "D": R(f"v2_{L}")["D"], "D_base": R(f"base_{L}").get("D"),
    }
num["soft"]["primer"] = None
num["hard"]["primer"] = fin("primer_hard")
num["soft"]["pga100k"] = fin("pga100k_soft")
num["teacher"] = R("v2_soft")["teacher_acc"]
json.dump(num, open(f"{ROOT}/report/numbers.json", "w"), indent=1)
print(json.dumps({L: {k: v for k, v in num[L].items() if k not in ("pga", "pga_nodiv", "pga_lam1", "rounds", "D", "D_base")}
                  for L in ("soft", "hard")}, indent=1))

# ---------------------------------------------------------------- Fig 1: pipeline
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, txt, fc, ec="#333", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else None)


def arr(x0, y0, x1, y1, txt="", c="#333", ls="-", off=(0, 1.2)):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, ls=ls))
    if txt:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], txt, ha="center", fontsize=7.5, color=c)


box(1, 33, 15, 9, "Noise generator\n(DIP-style priors)\n+ RandAugment", "#eef3fb")
box(24, 33, 15, 9, "TEACHER\nResNet-34 (82.2%)\nblack box", "#fbe9e9", ec=C_LIT, bold=True)
box(45.5, 33, 20, 9, "D = noise + teacher labels\nsoft: probabilities\nhard: one-hot", "#f3f3f3")
box(72, 33, 25, 9, "Stage A / C: K = 3 students\nResNet-18, KL (soft) or CE (hard)\nclass-balanced weights", "#e8f4ea")
arr(16.5, 37.5, 23.5, 37.5, "seed 50k")
arr(39.5, 37.5, 45, 37.5, "labels")
arr(66.5, 37.5, 71.5, 37.5)
box(1, 17, 15, 9, "Free noise pool\n1.6M images\n(no teacher)", "#eef3fb")
box(23.5, 17, 24.5, 9, r"Stage B: selection (4 rounds)" "\n" r"score $=z(\log\bar p_c)+0.5\,z(\mathrm{JS})$" "\n" r"class quotas from label deficit", "#fff6e5")
arr(16.8, 21.5, 22.8, 21.5)
arr(84, 32.5, 46.5, 24, "students score the pool (free)", c="#555", off=(-2, 2.6))
arr(35, 26.5, 31.5, 32.5, "200k chosen\nnoise images", c=C_LIT, off=(-6.5, -1.5))
box(52.5, 3, 23.5, 9, r"Stage D: PGA on the students" "\n" r"$J=\ln s_c+\lambda\,\mathrm{SKL}(r_i,r_j)$" "\n" r"$+\,\lambda\,\min_o\,\mathrm{KL}(s_{new}\,\|\,s_o)$", "#f1ecf8")
box(80.5, 3, 18.5, 9, "Stage E: final single\nResNet-18, trained on\nteacher + ensemble labels", "#e8f4ea")
arr(88, 32.5, 70, 12.5, "ensemble", c="#555", off=(6, 0))
arr(76.8, 7.5, 80.5, 7.5)
ax.text(40, 10.5, "PGA images: labelled by the students,\nNEVER shown to the teacher", fontsize=8, color=C_LIT,
        ha="center")
ax.text(1, 3.5, "Teacher queries: 50k seed + 200k selected = 250k (hard cap).\n"
                "Student queries (pool scoring, PGA, ensemble labels) are free.", fontsize=8, color="#333")
save(fig, f"{FIG}/fig1_pipeline.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: example images
SA = torch.load(f"{E2}/v2_soft/stageA_seed.pt", weights_only=False)
SB = torch.load(f"{E2}/v2_soft/stageB_selected.pt", weights_only=False)
SD = torch.load(f"{E2}/v2_soft/stageD_pga.pt", weights_only=False)
fig, axs = plt.subplots(3, 8, figsize=(10, 4.4))
g = torch.Generator().manual_seed(0)
rows = [("random noise seed\n(teacher label)", SA["X"], SA["TLP"]), ("selected by students\n(teacher label)", SB["X"], SB["TLP"]),
        ("PGA images\n(student label)", SD["X"], SD["L"])]
for i, (title, X, Lp) in enumerate(rows):
    idx = torch.randperm(len(X), generator=g)[:8]
    for j, k in enumerate(idx.tolist()):
        a = axs[i, j]
        a.imshow(X[k].permute(1, 2, 0).numpy()); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        p = Lp[k].exp()
        a.set_title(f"{NAMES[int(p.argmax())]} {p.max():.2f}", fontsize=7)
        for s in a.spines.values():
            s.set_visible(False)
    axs[i, 0].set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center")
save(fig, f"{FIG}/fig2_examples.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: budget curve
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for a, L in zip(axs, ("soft", "hard")):
    n = num[L]
    xs = [50, 100, 250, 500]
    ys = [n["b50k"], n["b100k"], n["v2_final"][0], n["b500k"]]
    a.plot(xs, ys, "o-", c=C_V2, lw=2, label="ours (v2), single student")
    a.errorbar([250], [n["v2_final"][0]], yerr=[n["v2_final"][1]], fmt="none", ecolor=C_V2, capsize=4, elinewidth=1.5)
    ye = [n["b50k_ens"], n["b100k_ens"], n["v2_ens"][0], n["b500k_ens"]]
    a.plot(xs, ye, "s--", c=C_V2, alpha=0.55, label="ours (v2), ensemble of 3")
    a.plot([250, 500], [n["base"][0], n["base500k"]], "o-", c=C_BASE, lw=2, label="baseline: random noise, 1 student")
    a.plot([250], [n["prof"]], "D", c=C_PROF, ms=7, label="professor's method (1 student + PGA)")
    for x, y in zip(xs, ys):
        a.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(7, -11), fontsize=8, c=C_V2)
    a.annotate(f"{n['base500k']:.1f}", (500, n["base500k"]), textcoords="offset points", xytext=(-4, -13), fontsize=8, c="#666")
    a.set_xscale("log"); a.set_xticks(xs); a.set_xticklabels([f"{x}k" for x in xs]); a.minorticks_off()
    a.set_xlabel("teacher queries (noise images labelled by the teacher)"); a.set_ylabel("CIFAR-100 test accuracy (%)")
    a.set_title(f"({'a' if L == 'soft' else 'b'}) {L.capitalize()} labels")
axs[0].legend(fontsize=7.5, loc="lower right")
save(fig, f"{FIG}/fig3_budget.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: class imbalance of the teacher's answers
TB = torch.load(f"{E2}/base_soft/stageA_seed.pt", weights_only=False)["TLP"]
TD = torch.cat([SA["TLP"], SB["TLP"]])
hb = np.sort(np.bincount(TB.argmax(1).numpy(), minlength=100))[::-1]
hd = np.sort(np.bincount(TD.argmax(1).numpy(), minlength=100))[::-1]
fig, a = plt.subplots(figsize=(6.4, 3.4))
a.bar(np.arange(100) - 0.2, np.maximum(hb, 0.8), width=0.45, color=C_BASE, label=f"250k random noise ({(hb < 100).sum()} classes < 100 images)")
a.bar(np.arange(100) + 0.2, np.maximum(hd, 0.8), width=0.45, color=C_V2, label=f"250k selected by the students ({(hd < 100).sum()} classes < 100)")
a.axhline(2500, ls="--", c="k", lw=1); a.text(70, 2900, "perfectly balanced = 2,500 per class", fontsize=8)
a.set_yscale("log"); a.set_xlabel("classes, sorted by number of teacher labels"); a.set_ylabel("images per class")
a.legend(fontsize=8)
save(fig, f"{FIG}/fig4_imbalance.png"); plt.close(fig)
num["imbalance"] = {"random_top10": float(hb[:10].sum() / hb.sum() * 100), "sel_top10": float(hd[:10].sum() / hd.sum() * 100),
                    "random_lt100": int((hb < 100).sum()), "sel_lt100": int((hd < 100).sum()),
                    "random_zero": int((hb == 0).sum()), "sel_zero": int((hd == 0).sum())}

# ---------------------------------------------------------------- Fig 5: selection rounds
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for L, c in (("soft", C_V2), ("hard", C_LIT)):
    n = num[L]
    q = [50] + [r["teacher_queries"] / 1000 for r in n["rounds"][:-1]] + [250]
    acc = [n["A_ens"]] + [r["ensemble_acc_after"] for r in n["rounds"][:-1]] + [R(f"v2_{L}")["C_ensemble"]["acc"]]
    axs[0].plot(q, acc, "o-", c=c, label=f"{L} labels")
    axs[1].plot([r["round"] for r in n["rounds"]], [r["teacher_label_is_intended_class"] for r in n["rounds"]], "o-", c=c,
                label=f"{L}: teacher gives the intended class")
axs[0].set_xlabel("teacher queries so far (k)"); axs[0].set_ylabel("ensemble test accuracy (%)")
axs[0].set_title("(a) Ensemble accuracy during selection"); axs[0].legend(fontsize=8)
axs[1].set_xlabel("selection round"); axs[1].set_ylabel("%"); axs[1].set_xticks([1, 2, 3, 4])
axs[1].axhline(1, ls=":", c="k"); axs[1].text(1.05, 2, "chance = 1%", fontsize=8)
axs[1].set_ylim(0, 25); axs[1].set_title("(b) Teacher returns the targeted class"); axs[1].legend(fontsize=8)
save(fig, f"{FIG}/fig5_rounds.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 6: ablations
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
build = [("baseline\n1 student", "base"), ("+ PGA\n(professor)", "prof"), ("3 students\nno balance", "nobal"),
         ("selection\n1 student", "one"), ("v2\n(all parts)", "v2_final")]
w = 0.38
for k, L in enumerate(("soft", "hard")):
    vals = [num[L][key][0] if isinstance(num[L][key], tuple) else num[L][key] for _, key in build]
    errs = [num[L][key][1] if isinstance(num[L][key], tuple) else 0 for _, key in build]
    b = axs[0].bar(np.arange(len(build)) + (k - 0.5) * w, vals, w, yerr=errs, capsize=3,
                   color=[C_V2, C_LIT][k], alpha=0.85, label=f"{L}")
    axs[0].bar_label(b, fmt="%.1f", fontsize=7, padding=2)
axs[0].set_xticks(range(len(build))); axs[0].set_xticklabels([b[0] for b in build], fontsize=7.5)
axs[0].set_ylabel("test accuracy (%)"); axs[0].set_title("(a) Contribution of each component"); axs[0].legend(fontsize=8)
adds = [("no\nPGA", "nopga"), ("λ = 0", "nodiv"), ("λ = 1", "lam1"),
        ("SimCLR\ninit", "sim"), ("hard-label\nsoftening", "primer"), ("CutMix", "mix")]
sd = {"soft": num["soft"]["v2_final"][1], "hard": num["hard"]["v2_final"][1]}
for k, L in enumerate(("soft", "hard")):
    d = [(num[L][key] - num[L]["v2s1"]) if num[L].get(key) is not None else np.nan for _, key in adds]
    b = axs[1].bar(np.arange(len(adds)) + (k - 0.5) * w, d, w, color=[C_V2, C_LIT][k], alpha=0.85, label=L)
    axs[1].bar_label(b, labels=[("" if np.isnan(v) else f"{v:+.1f}") for v in d], fontsize=7, padding=2)
    axs[1].axhspan(-sd[L], sd[L], color=[C_V2, C_LIT][k], alpha=0.07)
axs[1].axhline(0, c="k", lw=0.8)
axs[1].set_xticks(range(len(adds))); axs[1].set_xticklabels([a[0] for a in adds], fontsize=7.5)
axs[1].set_ylabel("change w.r.t. v2 (points)"); axs[1].set_title("(b) Modifications on top of v2")
axs[1].legend(fontsize=8)
save(fig, f"{FIG}/fig6_ablations.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 7: what the diversity term does (far points)
dev = "cuda" if torch.cuda.is_available() else "cpu"
studs = []
for k in range(3):
    m = build_model("resnet18"); load_weights(m, f"{E2}/v2_soft/student_C_{k}.pth"); studs.append(m.to(dev).eval())


def ens_lp(X):
    lps = torch.stack([predict_logprobs(m, X, dev) for m in studs])
    return torch.logsumexp(lps, 0) - math.log(3)


runs = [("λ = 0 (no diversity)", f"{E2}/nodiv_soft"), ("λ = 0.1 (v2)", f"{E2}/v2_soft"), ("λ = 1.0", f"{E3}/lam1_soft")]
P = {}
for name, d in runs:
    S = torch.load(f"{d}/stageD_pga.pt", weights_only=False)
    lp = ens_lp(S["X"]); y = lp.argmax(1)
    P[name] = (S["X"], lp, y)


def r_of(lp, y):                                  # softmax over the 99 non-target classes
    z = lp.clone(); z[torch.arange(len(z)), y] = -float("inf")
    return F.log_softmax(z, 1)


def mean_skl(r):
    p = r.exp(); r = torch.nan_to_num(r, neginf=0.0)
    kl = (p * r).sum(1, keepdim=True) - p @ r.t()
    s = 0.5 * (kl + kl.t()); n = len(r)
    return float((s.sum() - s.diag().sum()) / (n * (n - 1)))


skl = {name: [] for name, _ in runs}
for c in range(100):
    for name, _ in runs:
        X, lp, y = P[name]
        sel = (y == c).nonzero().squeeze(1)
        if len(sel) >= 5:
            skl[name].append(mean_skl(r_of(lp[sel], y[sel])))
num["diversity"] = {name: {"mean_pairwise_SKL_same_class": float(np.mean(v)), "kept": int(len(P[name][0]))}
                    for name, v in skl.items()}
cls = max(range(100), key=lambda c: min(int((P[nm][2] == c).sum()) for nm, _ in runs))
fig = plt.figure(figsize=(10, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
allr = torch.cat([r_of(P[nm][1], P[nm][2])[P[nm][2] == cls] for nm, _ in runs]).exp().numpy()
mu = allr.mean(0); U, Sv, Vt = np.linalg.svd(allr - mu, full_matrices=False)
lim = None
for i, (nm, _) in enumerate(runs):
    X, lp, y = P[nm]
    sel = (y == cls).nonzero().squeeze(1)
    pts = (r_of(lp[sel], y[sel]).exp().numpy() - mu) @ Vt[:2].T
    a = fig.add_subplot(gs[0, i])
    a.scatter(pts[:, 0], pts[:, 1], s=14, c=[C_BASE, C_V2, C_LIT][i], alpha=0.75)
    a.set_title(f"{nm}\n{len(sel)} images, mean SKL {mean_skl(r_of(lp[sel], y[sel])):.2f}", fontsize=8.5)
    a.set_xlabel("PC 1"); a.set_ylabel("PC 2") if i == 0 else None
    if lim is None:
        lim = (np.abs((allr - mu) @ Vt[:2].T).max(0) * 1.1)
    a.set_xlim(-lim[0], lim[0]); a.set_ylim(-lim[1], lim[1])
    sub = fig.add_subplot(gs[1, i]); sub.axis("off")
    k = sel[:16]
    grid = X[k].permute(0, 2, 3, 1).numpy()
    grid = np.concatenate([np.concatenate(list(grid[r * 4:(r + 1) * 4]), 1) for r in range(min(4, len(grid) // 4))], 0)
    sub.imshow(grid)
save(fig, f"{FIG}/fig7_diversity.png"); plt.close(fig)
num["diversity"]["class"] = NAMES[cls]
json.dump(num, open(f"{ROOT}/report/numbers.json", "w"), indent=1)
print("diversity:", num["diversity"]); print("imbalance:", num["imbalance"])
print("figures in", FIG)


# ---------------------------------------------------------------- Fig 8: per-class accuracy vs. number of teacher labels
from query_kd.data import load_cifar100  # noqa: E402
Xt, yt = load_cifar100(f"{ROOT}/cifar-100-python", "test")


def per_class_acc(path):
    m = build_model("resnet18"); load_weights(m, path); m = m.to(dev).eval()
    pred = predict_logprobs(m, Xt, dev).argmax(1)
    return np.array([(pred[yt == c] == c).float().mean().item() * 100 for c in range(100)])


acc_b = per_class_acc(f"{E2}/base_soft/student_final.pth")
acc_v = per_class_acc(f"{E2}/v2_soft/student_final.pth")
cnt_b = np.bincount(TB.argmax(1).numpy(), minlength=100)
cnt_v = np.bincount(TD.argmax(1).numpy(), minlength=100)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
axs[0].scatter(np.maximum(cnt_b, 0.8), acc_b, s=16, c=C_BASE, alpha=0.8, label=f"baseline (Spearman ρ = {spearman(cnt_b, acc_b):.2f})")
axs[0].scatter(np.maximum(cnt_v, 0.8), acc_v, s=16, c=C_V2, alpha=0.8, label=f"v2 (Spearman ρ = {spearman(cnt_v, acc_v):.2f})")
axs[0].set_xscale("log"); axs[0].set_xlabel("teacher labels for the class (log scale)"); axs[0].set_ylabel("per-class test accuracy (%)")
axs[0].set_title("(a) Accuracy follows the number of labels"); axs[0].legend(fontsize=8, loc="upper left")
order = np.argsort(acc_v - acc_b)
d = (acc_v - acc_b)[order]
axs[1].bar(range(100), d, color=np.where(d >= 0, C_V2, C_LIT), width=0.85)
axs[1].axhline(0, c="k", lw=0.8)
axs[1].set_xlabel("classes, sorted by gain"); axs[1].set_ylabel("v2 − baseline (points)")
axs[1].set_title(f"(b) Per-class gain: {int((d > 0).sum())} classes improve, {int((d < 0).sum())} degrade")
save(fig, f"{FIG}/fig8_perclass.png"); plt.close(fig)


def H(c):
    p = c / c.sum(); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


num["perclass"] = {"spearman_base": spearman(cnt_b, acc_b), "spearman_v2": spearman(cnt_v, acc_v),
                   "zero_acc_base": int((acc_b == 0).sum()), "zero_acc_v2": int((acc_v == 0).sum()),
                   "lt10_base": int((acc_b < 10).sum()), "lt10_v2": int((acc_v < 10).sum()),
                   "improved": int((acc_v > acc_b).sum()), "degraded": int((acc_v < acc_b).sum()),
                   "median_gain": float(np.median(acc_v - acc_b))}
num["entropy_bits"] = {"random": H(cnt_b), "selected": H(cnt_v), "uniform": float(np.log2(100)),
                       "eff_classes_random": 2 ** H(cnt_b), "eff_classes_selected": 2 ** H(cnt_v)}

# ---------------------------------------------------------------- table of every run (appendix)
rows = []
for d in (E2, E3):
    for name in sorted(os.listdir(d)):
        p = f"{d}/{name}/results.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p)); a = json.load(open(f"{d}/{name}/args.json"))
        if "final_single_student" not in r:
            continue
        cs = [v["acc"] for v in r.get("C_students", {}).values()]
        f = r["final_single_student"]; e = r.get("C_ensemble", f)
        rows.append((a["label_mode"], name, a["n_students"], a["total_budget"], a.get("seed", 1),
                     (f"{min(cs):.1f}–{max(cs):.1f}" if len(cs) > 1 else f"{cs[0]:.1f}") if cs else "–",
                     e["acc"], f["acc"], f["top5"], f["agreement"]))
rows.sort(key=lambda t: (t[0], t[3], t[1]))
with open(f"{ROOT}/report/all_runs.md", "w") as fh:
    fh.write("| Run | Labels | K | Queries | Seed | Stage-C students | Ensemble | Final | Final top-5 | Agreement |\n")
    fh.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for t in rows:
        fh.write(f"| `{t[1]}` | {t[0]} | {t[2]} | {t[3] // 1000}K | {t[4]} | {t[5]} | {t[6]:.2f} | {t[7]:.2f} | {t[8]:.2f} | {t[9]:.2f} |\n")
num["final_top5"] = {L: R(f"v2_{L}")["final_single_student"]["top5"] for L in ("soft", "hard")}
json.dump(num, open(f"{ROOT}/report/numbers.json", "w"), indent=1)
print("perclass:", num["perclass"]); print("entropy:", num["entropy_bits"]); print("runs:", len(rows))
