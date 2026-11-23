#!/usr/bin/env python3
"""Contrastive pre-training (SimCLR) of the student ResNet-18 on FREE noise images - no teacher, no real data.

Idea from DIP-KD (contrastive learning among synthetic samples): before distillation, the student learns features
that tell noise images apart (two augmented views of the same image must match, all other images must differ).
The resulting weights are used as --init_ckpt for the students in run_multi_v2.py.

  loss = NT-Xent(z1, z2),  z = g(f(aug(x))),  f = ResNet-18 without classifier, g = 2-layer projection head
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from query_kd.data import RandAugmenter, normalize  # noqa: E402
from query_kd.models import build_model, save_weights  # noqa: E402
from query_kd.noise import make_noise  # noqa: E402
from query_kd.oracle import amp_ctx  # noqa: E402


def augment(x):
    """SimCLR augmentations on GPU for float images in [0,1]: random resized crop + flip, colour jitter, grayscale."""
    n, dev = len(x), x.device
    area = torch.empty(n, device=dev).uniform_(0.2, 1.0)
    logr = torch.empty(n, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    sw = (area * logr.exp()).sqrt().clamp(max=1.0)
    sh = (area / logr.exp()).sqrt().clamp(max=1.0)
    tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - sw)
    ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - sh)
    flip = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = sw * flip, tx, sh, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    jit = (torch.rand(n, 1, 1, 1, device=dev) < 0.8).float()
    b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.8 - 0.4) * jit
    c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.8 - 0.4) * jit
    s = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 0.8 - 0.4) * jit
    x = x * b
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]).expand_as(x)
    x = g + (x - g) * s
    gray = torch.rand(n, 1, 1, 1, device=dev) < 0.2
    x = torch.where(gray, g, x)
    return x.clamp(0, 1)


def nt_xent(z1, z2, tau):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / tau
    n = len(z1)
    sim.fill_diagonal_(-float("inf"))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output checkpoint (.pth), loadable with build_model + load_weights")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--n_images", type=int, default=500000, help="Free noise images (new set every 'refresh' epochs)")
    ap.add_argument("--refresh", type=int, default=5, help="Generate a fresh noise set every this many epochs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--noise_kinds", default="dip")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log_f = open(os.path.splitext(args.out)[0] + "_log.txt", "a")

    def log(msg):
        msg = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(msg, flush=True); log_f.write(msg + "\n"); log_f.flush()

    ra = RandAugmenter(2, 9)

    def noise(n):
        return torch.cat([ra.batched(make_noise(min(10000, n - i), args.noise_kinds, device=dev), dev)
                          for i in range(0, n, 10000)])

    model = build_model(args.arch).to(dev)
    fc = model.fc
    model.fc = nn.Identity()
    head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128)).to(dev)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    spe = args.n_images // args.batch_size
    total, warm = args.epochs * spe, spe
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")
    log(f"SimCLR on free noise: {args.n_images} images (refreshed every {args.refresh} epochs), {args.epochs} epochs, "
        f"batch {args.batch_size}, lr {args.lr}, tau {args.tau}")
    t0 = time.time()
    X = None
    for ep in range(args.epochs):
        if ep % args.refresh == 0:
            del X
            X = noise(args.n_images).to(dev)
        model.train(); head.train()
        perm = torch.randperm(len(X), device=dev)
        tot = 0.0
        for st in range(spe):
            xb = X[perm[st * args.batch_size:(st + 1) * args.batch_size]].float().div_(255.0)
            with amp_ctx(dev):
                z1 = head(model(normalize(augment(xb))))
                z2 = head(model(normalize(augment(xb))))
            loss = nt_xent(z1.float(), z2.float(), args.tau)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() if st % 50 == 0 else 0.0
        log(f"  epoch {ep + 1}/{args.epochs}  loss {tot / max(1, (spe + 49) // 50):.4f}  ({time.time() - t0:.0f}s)")
        model.fc = fc
        save_weights(model, args.out, epoch=ep + 1)
        model.fc = nn.Identity()
    log(f"Done. Saved {args.out}")


if __name__ == "__main__":
    main()
