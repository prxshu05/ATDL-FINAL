"""Knowledge-distillation training and evaluation (all data lives on the GPU for speed)."""

import math
import time

import torch
import torch.nn.functional as F

from .data import gpu_crop_flip, normalize
from .oracle import amp_ctx, predict_logprobs


def kd_loss(student_logits, teacher_logp, mode: str, T: float, alpha: float, reduction: str = "mean"):
    """mode: 'soft'  -> T^2 * KL(softmax(t/T) || softmax(s/T))       (Hinton KD)
             'hard'  -> CE(s, argmax t)                              (teacher hard labels only)
             'mse'   -> MSE between centred logits (ZOPGA-KD, Kim et al. 2021)
             'combined' -> alpha * soft + (1 - alpha) * hard
    teacher_logp are log-probs; softmax(logp / T) == softmax(logits / T), so any T works."""
    hard = teacher_logp.argmax(1)
    ce = F.cross_entropy(student_logits, hard, reduction="none")
    if mode == "hard":
        per = ce
    elif mode == "mse":  # ZOPGA-KD: MSE between logits; log-probs are logits up to a per-sample shift -> centre both
        t = teacher_logp - teacher_logp.mean(1, keepdim=True)
        s = student_logits - student_logits.mean(1, keepdim=True)
        per = ((s - t) ** 2).mean(1)
    else:
        soft = F.kl_div(F.log_softmax(student_logits / T, 1), F.log_softmax(teacher_logp / T, 1),
                        reduction="none", log_target=True).sum(1) * (T * T)
        per = soft if mode == "soft" else alpha * soft + (1 - alpha) * ce
    return per.mean() if reduction == "mean" else per


def train_student(model, X: torch.Tensor, TLP: torch.Tensor, epochs: int, lr: float, device: str,
                  label_mode: str = "soft", T: float = 4.0, alpha: float = 0.9, batch_size: int = 128,
                  weight_decay: float = 5e-4, warmup_epochs: float = 1.0, log=print, tag: str = "", index=None,
                  hard_mask=None, weight=None, mix: str = "none", mix_prob: float = 0.5, mix_alpha: float = 1.0):
    """X: uint8 (N,3,32,32) on device. TLP: target log-probs (N,100) on device.
    mix: 'none' | 'mixup' | 'cutmix' - with probability mix_prob a batch is mixed with a shuffled copy of itself
    (lam ~ Beta(mix_alpha, mix_alpha)); loss = lam * loss(target_a) + (1 - lam) * loss(target_b). Free: no teacher query.
    index: optional tensor of sample indices defining one epoch (allows oversampling); default all of X.
    hard_mask: optional bool (N,) - these samples use plain CE on argmax (student pseudo-labels), others label_mode.
    weight: optional float (N,) per-sample loss weight (e.g. < 1 for student-labelled PGA images).
    SGD + Nesterov, linear warm-up then cosine decay to 0 (fresh schedule on every call)."""
    model.train()
    if index is None:
        index = torch.arange(len(X), device=device)
    index = index.to(device)
    n = len(index)
    bs = min(batch_size, n)
    steps_per_epoch = max(1, n // bs)  # drop last partial batch (BatchNorm-safe)
    total = epochs * steps_per_epoch
    warm = int(warmup_epochs * steps_per_epoch)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    t0 = time.time()
    for ep in range(epochs):
        perm = index[torch.randperm(n, device=device)]
        run_loss = 0.0
        for st in range(steps_per_epoch):
            idx = perm[st * bs:(st + 1) * bs]
            xb = gpu_crop_flip(X[idx].float().div_(255.0))
            idx_b, lam = None, 1.0
            if mix != "none" and torch.rand(1).item() < mix_prob:
                idx_b = idx[torch.randperm(len(idx), device=idx.device)]
                xb2 = gpu_crop_flip(X[idx_b].float().div_(255.0))
                lam = float(torch.distributions.Beta(mix_alpha, mix_alpha).sample())
                if mix == "mixup":
                    xb = lam * xb + (1 - lam) * xb2
                else:                                                       # cutmix: paste a box of xb2
                    H, W = xb.shape[-2:]
                    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
                    cy, cx = torch.randint(H, (1,)).item(), torch.randint(W, (1,)).item()
                    y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
                    x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
                    xb = xb.clone(); xb[:, :, y0:y1, x0:x1] = xb2[:, :, y0:y1, x0:x1]
                    lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
            xb = normalize(xb)
            with amp_ctx(device):
                logits = model(xb)

            def per_sample(ix):
                per = kd_loss(logits.float(), TLP[ix].float(), label_mode, T, alpha, reduction="none")
                if hard_mask is not None:
                    ce = F.cross_entropy(logits.float(), TLP[ix].argmax(1), reduction="none")
                    per = torch.where(hard_mask[ix], ce, per)
                if weight is not None:
                    per = per * weight[ix]
                return per

            if idx_b is None and weight is None and hard_mask is None:
                loss = kd_loss(logits.float(), TLP[idx].float(), label_mode, T, alpha)
            elif idx_b is None:
                loss = per_sample(idx).mean()
            else:
                loss = (lam * per_sample(idx) + (1 - lam) * per_sample(idx_b)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() if st == steps_per_epoch - 1 else 0.0
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            log(f"    {tag} epoch {ep + 1}/{epochs}  loss {run_loss:.4f}  lr {sched.get_last_lr()[0]:.4f}  "
                f"({time.time() - t0:.0f}s)")
    model.eval()


@torch.no_grad()
def evaluate(model, X_test: torch.Tensor, y_test: torch.Tensor, device: str, teacher_test_lp: torch.Tensor = None) -> dict:
    """Accuracy / top-5 on CIFAR-100 test, plus fidelity to the teacher (agreement + KL(T||S))."""
    lp = predict_logprobs(model, X_test, device)
    res = {
        "acc": (lp.argmax(1) == y_test).float().mean().item() * 100,
        "top5": (lp.topk(5, 1).indices == y_test.view(-1, 1)).any(1).float().mean().item() * 100,
    }
    if teacher_test_lp is not None:
        res["agreement"] = (lp.argmax(1) == teacher_test_lp.argmax(1)).float().mean().item() * 100
        res["kl_teacher_student"] = (teacher_test_lp.exp() * (teacher_test_lp - lp)).sum(1).mean().item()
    return res
