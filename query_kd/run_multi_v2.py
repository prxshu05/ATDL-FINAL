#!/usr/bin/env python3
"""Version 2 of run_multi.py (several students, the teacher sees only noise). Changes from version 1:
  * Stage B is done in ROUNDS (--sel_rounds): choose part of the budget, query the teacher, fine-tune the students on
    the new data, choose the next part with the better students (active learning).
  * Selection score uses z-scores, so student disagreement really counts:
        score_c(x) = z( log p_mean_c(x) ) + dis_weight * z( D(x) )
  * Class quotas follow the DEFICIT: half of each round is split equally over the classes, half goes to the classes
    that still have the fewest teacher labels.
  * --lam_intra / --lam_old set the PGA diversity weights (0 = ablation without the diversity formula).
  * --reuse_from copies stages A-C of another run (for ablations that only change stages D-E).

  Stage A  Seed:      seed_size noise images + RandAugment -> TEACHER -> D. Train K students (ResNet-18, different
                      random init) on D.
  Stage B  Selection: pool_size free noise images. The K students (free) score them. The remaining teacher budget is
                      spent on images chosen CLASS-BALANCED (an equal quota per class, ranked by the students' mean
                      probability for that class) and preferring images the students DISAGREE on (DisGUIDE idea).
                      -> TEACHER -> added to D.  The teacher still sees nothing but noise images.
  Stage C  Students:  K fresh students trained on all of D with class-balanced loss weights (rare classes count more).
  Stage D  PGA:       PGA on the ENSEMBLE of the K students (target class c, confidence 0.8/0.9, KL diversity).
                      Labelled by the ensemble (free); kept only when ALL K students give the target class and the
                      ensemble confidence >= pga_min_conf.
  Stage E  Final:     one fresh ResNet-18 learns from teacher labels on D + ensemble labels on the PGA images and on
                      extra free noise images (distillation of the ensemble into a single student).

Every stage is saved to disk; running the same command again continues after the last finished stage.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from query_kd.data import RandAugmenter, load_cifar100  # noqa: E402
from query_kd.models import build_model, load_weights, save_weights  # noqa: E402
from query_kd.noise import make_noise  # noqa: E402
from query_kd.oracle import TeacherOracle, predict_logprobs  # noqa: E402
from query_kd.pga_ascent import PGAGenerator  # noqa: E402
from query_kd.trainer import evaluate, train_student  # noqa: E402


class Ensemble(nn.Module):
    """Average of the students' probabilities; forward returns log(mean softmax), usable as logits."""

    def __init__(self, models):
        super().__init__()
        self.ms = nn.ModuleList(models)

    def forward(self, x):
        lp = torch.stack([F.log_softmax(m(x).float(), 1) for m in self.ms])
        return torch.logsumexp(lp, 0) - math.log(len(self.ms))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--teacher_arch", default="resnet34")
    ap.add_argument("--student_arch", default="resnet18")
    ap.add_argument("--out_dir", default="experiments/multi_kd")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--label_mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--n_students", type=int, default=3)
    ap.add_argument("--seed_size", type=int, default=50000)
    ap.add_argument("--total_budget", type=int, default=250000, help="Teacher queries (seed + selected noise)")
    ap.add_argument("--pool_size", type=int, default=1500000, help="Free noise images scored by the students")
    ap.add_argument("--dis_weight", type=float, default=0.5, help="Weight of z-scored student disagreement in the selection")
    ap.add_argument("--sel_rounds", type=int, default=4, help="Selection rounds in stage B (students retrained between)")
    ap.add_argument("--round_epochs", type=int, default=10, help="Fine-tuning epochs of the students between rounds")
    ap.add_argument("--round_lr", type=float, default=0.05)
    ap.add_argument("--lam_intra", type=float, default=0.1, help="PGA same-class diversity weight (SKL)")
    ap.add_argument("--lam_old", type=float, default=0.1, help="PGA new-vs-old diversity weight (KL)")
    ap.add_argument("--final_init", choices=["scratch", "best_c"], default="best_c",
                    help="best_c = final student starts from the best stage-C student and is fine-tuned (v1: scratch)")
    ap.add_argument("--final_lr", type=float, default=0.02, help="LR for the final student when --final_init best_c")
    ap.add_argument("--reuse_from", default=None, help="Copy stages A-C (seed, selection, students) from this run dir")
    ap.add_argument("--reuse_stages", choices=["ABC", "AB"], default="ABC",
                    help="AB = copy only the teacher-labelled data (stages A-B); the stage-C students are trained again")
    ap.add_argument("--init_ckpt", default=None,
                    help="Start every new student from these weights (e.g. contrastive pre-training on noise); "
                         "the classifier layer is re-initialised per student")
    ap.add_argument("--mix", choices=["none", "mixup", "cutmix"], default="none",
                    help="Mix pairs of training images and their labels during student training (free, no teacher query)")
    ap.add_argument("--mix_prob", type=float, default=0.5)
    ap.add_argument("--hard_soften", type=float, default=1.0,
                    help="Hard labels only, stage E: teacher one-hot target -> beta*one-hot + (1-beta)*ensemble softmax "
                         "(primer-student idea of DIP-KD). 1.0 = off")
    ap.add_argument("--noise_kinds", default="dip")
    ap.add_argument("--randaug_ops", type=int, default=2)
    ap.add_argument("--randaug_magnitude", type=int, default=9)
    ap.add_argument("--balance_power", type=float, default=0.5,
                    help="Class-balanced loss weight = (mean count / class count)^power, clipped to [0.25, 4]")
    ap.add_argument("--pga_images", type=int, default=20000)
    ap.add_argument("--pga_steps", type=int, default=300)
    ap.add_argument("--pga_min_conf", type=float, default=0.8)
    ap.add_argument("--extra_noise", type=int, default=250000, help="Free noise images labelled by the ensemble")
    ap.add_argument("--ens_weight", type=float, default=0.5, help="Loss weight of ensemble-labelled images")
    ap.add_argument("--seed_epochs", type=int, default=60)
    ap.add_argument("--student_epochs", type=int, default=40)
    ap.add_argument("--final_epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    out = os.path.join(args.out_dir, args.run_name)
    os.makedirs(out, exist_ok=True)
    if args.reuse_from and not os.path.exists(os.path.join(out, "stageA_seed.pt")):
        import glob
        import shutil
        stud = "student_[AC]_*.pth" if args.reuse_stages == "ABC" else "student_A_*.pth"
        for f in glob.glob(os.path.join(args.reuse_from, "stage[AB]_*.pt")) + \
                glob.glob(os.path.join(args.reuse_from, stud)):
            dst = os.path.join(out, os.path.basename(f))
            try:
                os.link(f, dst)                                        # hard link: no extra disk space
            except OSError:
                shutil.copy(f, dst)
        r_src = json.load(open(os.path.join(args.reuse_from, "results.json")))
        drop = {"pga", "final_single_student", "final_train_size"}
        if args.reuse_stages == "AB":
            drop |= {"C_students", "C_ensemble"}
        json.dump({k: v for k, v in r_src.items() if k not in drop},
                  open(os.path.join(out, "results.json"), "w"), indent=2)
    log_file = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(msg, flush=True)
        log_file.write(msg + "\n"); log_file.flush()

    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=2)
    log(f"=== Run '{args.run_name}' [v2] ({args.label_mode} labels, {args.n_students} students, seed {args.seed}) "
        f"on {device} | budget {args.total_budget} (seed {args.seed_size}) | {args.sel_rounds} selection rounds | "
        f"PGA lam_intra {args.lam_intra} lam_old {args.lam_old} | class balance power {args.balance_power} | "
        f"mix {args.mix} | init {args.init_ckpt or 'random'} | hard_soften {args.hard_soften} ===")
    MIX = dict(mix=args.mix, mix_prob=args.mix_prob)

    X_test, y_test = load_cifar100(args.data_dir, "test")
    teacher = build_model(args.teacher_arch)
    load_weights(teacher, args.teacher_ckpt)
    oracle = TeacherOracle(teacher, device, budget=args.total_budget)
    teacher_test_lp = predict_logprobs(teacher, X_test, device)
    t_acc = (teacher_test_lp.argmax(1) == y_test).float().mean().item() * 100
    log(f"Teacher ({args.teacher_arch}) test accuracy {t_acc:.2f}% | teacher query budget {args.total_budget} "
        f"(teacher sees ONLY noise images)")
    ra = RandAugmenter(args.randaug_ops, args.randaug_magnitude)
    K = args.n_students
    results_path = os.path.join(out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {"teacher_acc": t_acc}

    def save_results():
        json.dump(results, open(results_path, "w"), indent=2)

    def noise(n):
        """n noise images + RandAugment, made 10k at a time to keep memory low."""
        chunks = []
        for i in range(0, n, 10000):
            xc = make_noise(min(10000, n - i), args.noise_kinds, device=device)
            chunks.append(ra.batched(xc, device) if device == "cuda" else ra(xc))
        return torch.cat(chunks)

    def ask_teacher(x):
        lp = oracle.query(x)
        if args.label_mode == "hard":
            lp = F.one_hot(lp.argmax(1), 100).float().clamp_min(1e-8).log()
        return lp

    def class_weights(TLP):
        cnt = torch.bincount(TLP.argmax(1), minlength=100).float().clamp_min(1)
        w = ((cnt.mean() / cnt) ** args.balance_power).clamp(0.25, 4.0)[TLP.argmax(1)]
        return w / w.mean()

    def coverage(lp):
        h = torch.bincount(lp.argmax(1).cpu(), minlength=100)
        top10 = 100.0 * h.sort(descending=True).values[:10].sum().item() / max(1, len(lp))
        return int((h > 0).sum()), int((h < 100).sum()), top10

    def ev(model, name):
        res = evaluate(model, X_test, y_test, device, teacher_test_lp)
        log(f"    {name}: test acc {res['acc']:.2f}% (top5 {res['top5']:.2f}%) | agree w/ teacher "
            f"{res['agreement']:.2f}% | KL(T||S) {res['kl_teacher_student']:.3f}")
        return res

    def train_students(tag, X, TLP, epochs, weight=None):
        models = []
        for k in range(K):
            path = os.path.join(out, f"student_{tag}_{k}.pth")
            m = build_model(args.student_arch).to(device)
            if os.path.exists(path):
                load_weights(m, path); m.eval()
                log(f"[{tag}] student {k}: loaded from disk")
            else:
                torch.manual_seed(args.seed * 100 + k)                 # different random init per student
                if args.init_ckpt:
                    load_weights(m, args.init_ckpt)
                    m.fc.reset_parameters()                            # own classifier per student
                log(f"[{tag}] training student {k + 1}/{K} on {len(X)} images for {epochs} epochs")
                train_student(m, X, TLP, epochs, args.lr, device=device, label_mode=args.label_mode,
                              T=args.temperature, log=log, tag=f"{tag}-s{k}", weight=weight, **MIX)
                save_weights(m, path)
                results.setdefault(f"{tag}_students", {})[str(k)] = ev(m, f"{tag} student {k}")
                save_results()
            models.append(m)
        ens = Ensemble(models).to(device).eval()
        if f"{tag}_ensemble" not in results:
            results[f"{tag}_ensemble"] = ev(ens, f"{tag} ENSEMBLE of {K} students")
            save_results()
        return models, ens

    # ------------------------------------------------------------------ Stage A: noise seed -> teacher
    pA = os.path.join(out, "stageA_seed.pt")
    if os.path.exists(pA):
        S = torch.load(pA, weights_only=False); X_seed, T_seed = S["X"], S["TLP"]; oracle.used = S["tq"]
        log(f"[A] seed loaded from disk ({len(X_seed)} images, teacher queries {oracle.used})")
    else:
        X_seed = noise(args.seed_size)
        T_seed = ask_teacher(X_seed)
        torch.save({"X": X_seed, "TLP": T_seed, "tq": oracle.used}, pA)
        n, rare, top10 = coverage(T_seed)
        log(f"[A] {len(X_seed)} noise images -> teacher | label classes present {n}/100, classes with <100 images "
            f"{rare}, top-10 classes hold {top10:.1f}% of the images")
    studA, ensA = train_students("A", X_seed.to(device), T_seed.to(device), args.seed_epochs)

    # ------------------------------------------------------------------ Stage B: class-balanced selection in rounds
    pB = os.path.join(out, "stageB_selected.pt")
    n_sel = args.total_budget - args.seed_size
    R = max(1, args.sel_rounds)
    if os.path.exists(pB):
        S = torch.load(pB, weights_only=False); X_sel, T_sel = S["X"], S["TLP"]; oracle.used = S["tq"]
        log(f"[B] selected images loaded from disk ({len(X_sel)}, teacher queries {oracle.used})")
    elif n_sel <= 0:
        X_sel, T_sel = torch.empty(0, 3, 32, 32, dtype=torch.uint8), torch.empty(0, 100)
        log("[B] no selection (the whole budget is the random seed)")
    else:
        X_sel, T_sel, r0 = torch.empty(0, 3, 32, 32, dtype=torch.uint8), torch.empty(0, 100), 0
        for r in range(R, 0, -1):
            pr = os.path.join(out, f"stageB_round{r}.pt")
            if os.path.exists(pr):
                S = torch.load(pr, weights_only=False)
                X_sel, T_sel, oracle.used, r0 = S["X"], S["TLP"], S["tq"], r
                for k, m in enumerate(studA):
                    load_weights(m, os.path.join(out, f"student_B{r}_{k}.pth")); m.eval()
                log(f"[B] resuming after selection round {r} ({len(X_sel)} selected, teacher queries {oracle.used})")
                break
        for r in range(r0, R):
            n_r = n_sel // R + (n_sel % R if r == R - 1 else 0)
            counts = torch.bincount(torch.cat([T_seed, T_sel]).argmax(1), minlength=100).float()
            target = (len(T_seed) + len(T_sel) + n_r) / 100.0
            deficit = (target - counts).clamp_min(0)
            share = 0.5 / 100 + 0.5 * (deficit / deficit.sum() if deficit.sum() > 0 else torch.full((100,), 0.01))
            quota = (share * n_r).floor().long()
            quota[torch.argsort(-share)[: n_r - int(quota.sum())]] += 1
            pool_n = max(args.pool_size // R, 4 * n_r)
            log(f"[B] round {r + 1}/{R}: scoring {pool_n:,} free noise images with the {K} students, choosing {n_r} "
                f"(quota per class {int(quota.min())}-{int(quota.max())})")
            pool, mean_lp, dis = [], [], []
            for i in range(0, pool_n, 50000):
                xc = noise(min(50000, pool_n - i))
                lps = torch.stack([predict_logprobs(m, xc, device) for m in studA])
                mlp = torch.logsumexp(lps, 0) - math.log(K)
                dis.append((lps.exp() * (lps - mlp[None])).sum(2).mean(0))
                pool.append(xc); mean_lp.append(mlp)
            pool, mean_lp, dis = torch.cat(pool), torch.cat(mean_lp), torch.cat(dis)
            z_lp = (mean_lp - mean_lp.mean(0)) / mean_lp.std(0).clamp_min(1e-6)
            z_d = (dis - dis.mean()) / dis.std().clamp_min(1e-6) if K > 1 else torch.zeros_like(dis)
            score = z_lp + args.dis_weight * z_d[:, None]
            taken = torch.zeros(len(pool), dtype=torch.bool)
            sel, intended = [], []
            for c in torch.argsort(counts).tolist():                       # classes with fewest labels pick first
                q = int(quota[c])
                if q == 0:
                    continue
                idx = score[:, c].masked_fill(taken, -float("inf")).topk(q).indices
                taken[idx] = True; sel.append(idx); intended.append(torch.full((q,), c))
            sel, intended = torch.cat(sel), torch.cat(intended)
            x_r = pool[sel].clone(); del pool, score, z_lp
            t_r = ask_teacher(x_r)
            hit = (t_r.argmax(1) == intended).float().mean().item() * 100
            X_sel, T_sel = torch.cat([X_sel, x_r]), torch.cat([T_sel, t_r])
            n, rare, top10 = coverage(torch.cat([T_seed, T_sel]))
            results.setdefault("selection_rounds", []).append(
                {"round": r + 1, "n": n_r, "teacher_label_is_intended_class": hit,
                 "disagreement_selected": dis[sel].mean().item(), "disagreement_pool": dis.mean().item(),
                 "classes_present": n, "classes_below_100": rare, "top10_share": top10, "teacher_queries": oracle.used})
            log(f"[B] round {r + 1}: {n_r} images -> teacher | teacher gives the intended class for {hit:.1f}% | "
                f"disagreement selected {dis[sel].mean():.3f} vs pool {dis.mean():.3f} | D so far: classes present "
                f"{n}/100, classes with <100 images {rare}, top-10 classes hold {top10:.1f}% | teacher queries {oracle.used}")
            if r < R - 1:                                                  # better students choose the next round
                X_r = torch.cat([X_seed, X_sel]).to(device); T_rr = torch.cat([T_seed, T_sel]).to(device)
                w_r = class_weights(T_rr)
                for k, m in enumerate(studA):
                    train_student(m, X_r, T_rr, args.round_epochs, args.round_lr, device=device,
                                  label_mode=args.label_mode, T=args.temperature, log=log, tag=f"B{r + 1}-s{k}", weight=w_r,
                                  **MIX)
                    save_weights(m, os.path.join(out, f"student_B{r + 1}_{k}.pth"))
                res = ev(Ensemble(studA).to(device).eval(), f"B round {r + 1} ENSEMBLE (fine-tuned)")
                results["selection_rounds"][-1]["ensemble_acc_after"] = res["acc"]
                del X_r, T_rr
            torch.save({"X": X_sel, "TLP": T_sel, "tq": oracle.used}, os.path.join(out, f"stageB_round{r + 1}.pt"))
            if r > 0 and os.path.exists(os.path.join(out, f"stageB_round{r}.pt")):
                os.remove(os.path.join(out, f"stageB_round{r}.pt"))
            save_results()
        torch.save({"X": X_sel, "TLP": T_sel, "tq": oracle.used}, pB)
        last = os.path.join(out, f"stageB_round{R}.pt")
        if os.path.exists(last):
            os.remove(last)
    X_D = torch.cat([X_seed, X_sel]).to(device)
    T_D = torch.cat([T_seed, T_sel]).to(device)
    n, rare, top10 = coverage(T_D)
    log(f"[B] D = {len(X_D)} teacher-labelled noise images | teacher queries {oracle.used} | classes present {n}/100, "
        f"classes with <100 images {rare}, top-10 classes hold {top10:.1f}%")
    results["D"] = {"size": len(X_D), "teacher_queries": oracle.used, "classes_present": n,
                    "classes_below_100": rare, "top10_share": top10}
    w_D = class_weights(T_D)

    # ------------------------------------------------------------------ Stage C: K students on D
    del studA, ensA
    studC, ensC = train_students("C", X_D, T_D, args.student_epochs, weight=w_D)

    # ------------------------------------------------------------------ Stage D: PGA on the ensemble
    pD = os.path.join(out, "stageD_pga.pt")
    if os.path.exists(pD):
        S = torch.load(pD, weights_only=False); X_pga, L_pga = S["X"], S["L"]
        log(f"[D] PGA images loaded from disk ({len(X_pga)} kept)")
    elif args.pga_images > 0:
        pga = PGAGenerator(steps=args.pga_steps, conf=[0.8, 0.9], batch_size=250, lam_intra=args.lam_intra,
                           lam_old=args.lam_old)
        targets = torch.arange(args.pga_images) % 100
        x0 = noise(args.pga_images)
        ref = predict_logprobs(ensC, X_D, device)
        x, st = pga.generate(ensC, x0, targets, device, noise_fn=noise, ref_lp=ref, ref_y=T_D.argmax(1).cpu(), log=log)
        lps = torch.stack([predict_logprobs(m, x, device) for m in studC])
        L = torch.logsumexp(lps, 0) - math.log(K)
        all_agree = (lps.argmax(2) == targets[None]).all(0)
        keep = all_agree & (L.max(1).values.exp() >= args.pga_min_conf)
        diag = predict_logprobs(teacher, x, device).argmax(1)            # diagnostic only, not a query
        t_ok = (diag[keep] == targets[keep]).float().mean().item() * 100 if keep.any() else 0.0
        X_pga, L_pga = x[keep], L[keep]
        torch.save({"X": X_pga, "L": L_pga}, pD)
        results["pga"] = {"generated": len(x), "kept": int(keep.sum()), "diag_teacher_agrees_with_kept": t_ok,
                          **{k: v for k, v in st.items()}}
        save_results()
        log(f"[D] kept {int(keep.sum())}/{len(x)} PGA images (all {K} students say the target class, ensemble conf >= "
            f"{args.pga_min_conf}) | [diagnostic only, not a query] teacher would agree for {t_ok:.1f}% of them")
    else:
        X_pga, L_pga = torch.empty(0, 3, 32, 32, dtype=torch.uint8), torch.empty(0, 100)

    # ------------------------------------------------------------------ Stage E: one student from teacher + ensemble
    pE = os.path.join(out, "student_final.pth")
    if not os.path.exists(pE):
        X_ex = noise(args.extra_noise) if args.extra_noise > 0 else torch.empty(0, 3, 32, 32, dtype=torch.uint8)
        L_ex = predict_logprobs(ensC, X_ex, device) if len(X_ex) else torch.empty(0, 100)
        X_all = torch.cat([X_D.cpu(), X_pga, X_ex]).to(device)
        T_all = torch.cat([T_D.cpu(), L_pga, L_ex]).to(device)
        n_ens = len(X_pga) + len(X_ex)
        w = torch.cat([w_D.cpu(), torch.full((n_ens,), args.ens_weight)]).to(device)
        # hard mode: teacher-labelled images use CE on the teacher's class; ensemble-labelled images use the ensemble's
        # probabilities (our own students, free). soft mode: KL everywhere.
        hard_mask = None
        mode = args.label_mode
        if args.label_mode == "hard" and args.hard_soften < 1.0:
            # primer idea: soften the teacher's one-hot answers with our own ensemble's probabilities (free)
            b = args.hard_soften
            ens_D = predict_logprobs(ensC, X_D, device).to(device).exp()
            T_all[: len(X_D)] = (b * T_D.exp() + (1 - b) * ens_D).clamp_min(1e-8).log()
            mode = "soft"
            log(f"[E] hard labels softened: {b} * teacher one-hot + {1 - b:.2f} * ensemble softmax (no extra teacher query)")
        elif args.label_mode == "hard":
            mode = "soft"
            hard_mask = torch.cat([torch.ones(len(X_D), dtype=torch.bool), torch.zeros(n_ens, dtype=torch.bool)]).to(device)
        log(f"[E] final single {args.student_arch}: {len(X_D)} teacher-labelled + {len(X_pga)} PGA + {len(X_ex)} extra "
            f"noise (ensemble-labelled, weight {args.ens_weight}) = {len(X_all)} images, {args.final_epochs} epochs")
        torch.manual_seed(args.seed * 1000)
        final = build_model(args.student_arch).to(device)
        lr = args.lr
        if args.final_init == "best_c":
            accs = {int(k): v["acc"] for k, v in results["C_students"].items()}
            kb = max(accs, key=accs.get)
            load_weights(final, os.path.join(out, f"student_C_{kb}.pth"))
            lr = args.final_lr
            log(f"[E] starting from the best stage-C student ({kb}, {accs[kb]:.2f}%), fine-tuning at lr {lr}")
        train_student(final, X_all, T_all, args.final_epochs, lr, device=device, label_mode=mode,
                      T=args.temperature, log=log, tag="final", weight=w, hard_mask=hard_mask, **MIX)
        save_weights(final, pE)
        results["final_single_student"] = ev(final, "FINAL single ResNet-18")
        results["final_train_size"] = len(X_all)
        save_results()

    log(f"\nDone. Teacher {t_acc:.2f}% | teacher queries {oracle.used} | "
        f"ensemble of {K}: {results['C_ensemble']['acc']:.2f}% | single final student: "
        f"{results['final_single_student']['acc']:.2f}%. Results in {out}")
    log_file.close()


if __name__ == "__main__":
    main()
