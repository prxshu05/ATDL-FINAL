#!/usr/bin/env python3
"""Step 1 - Train the Teacher (ResNet-34) on the real CIFAR-100 training set (normal supervised training, no PGA).

  python query_kd/train_teacher.py --data_dir cifar-100-python --arch resnet34 --epochs 200

ResNet-34 teacher -> ResNet-18 student is the standard pair in data-free KD papers (also used in ZOPGA-KD),
so results can be compared with existing work. 200 epochs = the standard CIFAR schedule (SGD 0.1, cosine, wd 5e-4).
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
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from query_kd.data import cutmix_boxes, gpu_crop_flip, load_cifar100, normalize  # noqa: E402
from query_kd.models import build_model, save_weights  # noqa: E402
from query_kd.oracle import amp_ctx  # noqa: E402
from query_kd.trainer import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train the CIFAR-100 teacher")
    ap.add_argument("--data_dir", required=True, help="Path to 'cifar-100-python' folder (or its parent)")
    ap.add_argument("--arch", default="resnet34")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--cutmix_p", type=float, default=0.5, help="Probability of CutMix per batch (0 = off)")
    ap.add_argument("--train_fraction", type=float, default=1.0,
                    help="Use only this fraction of CIFAR-100 train for the teacher (e.g. 0.5)")
    ap.add_argument("--val_size", type=int, default=0,
                    help="Hold out this many TRAINING images as a validation set; the checkpoint is then chosen by "
                         "validation accuracy and the test set is only reported (0 = old behaviour: chosen by test)")
    ap.add_argument("--out", default="./experiments/teacher/teacher_resnet34.pth")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    Xtr, ytr = load_cifar100(args.data_dir, "train")
    Xte, yte = load_cifar100(args.data_dir, "test")
    if args.train_fraction < 1.0:
        keep = torch.randperm(len(Xtr))[: int(len(Xtr) * args.train_fraction)]
        Xtr, ytr = Xtr[keep], ytr[keep]
    Xva = yva = None
    if args.val_size > 0:
        p = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(args.seed))
        Xva, yva = Xtr[p[:args.val_size]], ytr[p[:args.val_size]]
        Xtr, ytr = Xtr[p[args.val_size:]], ytr[p[args.val_size:]]
        print(f"Validation split: {len(Xva)} of the training images held out (checkpoint chosen on these, not on test)")
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    print(f"Device {device} | train {len(Xtr)} | test {len(Xte)} | arch {args.arch} | epochs {args.epochs}")

    model = build_model(args.arch).to(device).to(memory_format=torch.channels_last)
    n = len(Xtr)
    spe = n // args.batch_size
    total, warm = args.epochs * spe, args.warmup_epochs * spe
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log_path = os.path.splitext(args.out)[0] + "_log.json"
    history, best = [], 0.0
    for ep in range(args.epochs):
        model.train()
        t0, perm, tot_loss = time.time(), torch.randperm(n, device=device), 0.0
        for st in range(spe):
            idx = perm[st * args.batch_size:(st + 1) * args.batch_size]
            x = gpu_crop_flip(Xtr[idx].float().div_(255.0))
            y = ytr[idx]
            use_cm = random.random() < args.cutmix_p
            if use_cm:
                perm2 = torch.randperm(len(idx), device=device)
                x, lam = cutmix_boxes(x, x[perm2])
            x = normalize(x).contiguous(memory_format=torch.channels_last)
            with amp_ctx(device):
                logits = model(x)
            logits = logits.float()
            if use_cm:
                loss = (lam * F.cross_entropy(logits, y, reduction="none")
                        + (1 - lam) * F.cross_entropy(logits, y[perm2], reduction="none")).mean()
            else:
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item()
        res = evaluate(model, Xte, yte, device)
        row = {"epoch": ep + 1, "loss": tot_loss / spe, **res}
        sel = res["acc"]
        if Xva is not None:
            row["val_acc"] = sel = evaluate(model, Xva, yva, device)["acc"]
        history.append(row)
        tag = ""
        if sel > best:
            best = sel
            best_row = row
            save_weights(model, args.out, epoch=ep + 1, best_acc=best, arch=args.arch)
            tag = "  <- best, saved"
        print(f"Epoch {ep + 1:3d}/{args.epochs}  loss {tot_loss / spe:.4f}  "
              + (f"val acc {row['val_acc']:.2f}%  " if Xva is not None else "")
              + f"test acc {res['acc']:.2f}%  top5 {res['top5']:.2f}%  ({time.time() - t0:.0f}s){tag}", flush=True)
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
    save_weights(model, os.path.splitext(args.out)[0] + "_last.pth", epoch=args.epochs, arch=args.arch)
    if Xva is not None:
        print(f"Done. Checkpoint chosen by VALIDATION: epoch {best_row['epoch']} | val {best_row['val_acc']:.2f}% | "
              f"test {best_row['acc']:.2f}% | last epoch test {history[-1]['acc']:.2f}% -> {args.out}")
    else:
        print(f"Done. Best teacher test accuracy {best:.2f}% -> {args.out}")


if __name__ == "__main__":
    main()
