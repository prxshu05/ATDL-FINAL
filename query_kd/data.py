"""Data utilities: CIFAR-100 loading (teacher training + test set), GPU augmentation, RandAugment.

All images are kept as uint8 tensors of shape (N, 3, 32, 32) -- 100k images = ~300 MB.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# ----------------------------------------------------------------------------- CIFAR-100
def resolve_cifar_dir(path: str) -> str:
    """Accepts either the 'cifar-100-python' folder itself or its parent folder."""
    for cand in (path, os.path.join(path, "cifar-100-python")):
        if os.path.isfile(os.path.join(cand, "train")) and os.path.isfile(os.path.join(cand, "test")):
            return cand
    raise FileNotFoundError(
        f"Could not find CIFAR-100 'train' and 'test' files in '{path}' or '{path}/cifar-100-python'."
    )


def load_cifar100(path: str, split: str):
    """Returns (images uint8 (N,3,32,32), fine labels long (N,)) for split 'train' or 'test'."""
    d = resolve_cifar_dir(path)
    with open(os.path.join(d, split), "rb") as f:
        obj = pickle.load(f, encoding="bytes")
    x = torch.from_numpy(np.asarray(obj[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).copy())
    y = torch.tensor(obj[b"fine_labels"], dtype=torch.long)
    return x, y


# ----------------------------------------------------------------------------- tensors
def normalize(x_uint8_or_float: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] or float [0,1] -> normalized float."""
    x = x_uint8_or_float
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    mean = torch.tensor(CIFAR100_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def gpu_crop_flip(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Per-sample random crop (with reflect padding) + horizontal flip, fully on GPU. x: float (N,3,H,W)."""
    n, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (i.view(n, 1) + torch.arange(h, device=x.device)).view(n, 1, h, 1)
    cols = (j.view(n, 1) + torch.arange(w, device=x.device)).view(n, 1, 1, w)
    bidx = torch.arange(n, device=x.device).view(n, 1, 1, 1)
    cidx = torch.arange(3, device=x.device).view(1, 3, 1, 1)
    out = xp[bidx, cidx, rows, cols]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), out.flip(3), out)


def cutmix_boxes(a: torch.Tensor, b: torch.Tensor):
    """Pastes a random box of b into a (per-sample). Returns (mixed, lam) with lam = fraction kept from a."""
    n, _, h, w = a.shape
    dev = a.device
    lam = torch.rand(n, device=dev)
    cut_h = (h * torch.sqrt(1 - lam)).long()
    cut_w = (w * torch.sqrt(1 - lam)).long()
    cy = torch.randint(0, h, (n,), device=dev)
    cx = torch.randint(0, w, (n,), device=dev)
    y1 = (cy - cut_h // 2).clamp(0, h); y2 = (cy + cut_h // 2).clamp(0, h)
    x1 = (cx - cut_w // 2).clamp(0, w); x2 = (cx + cut_w // 2).clamp(0, w)
    ys = torch.arange(h, device=dev).view(1, h, 1)
    xs = torch.arange(w, device=dev).view(1, 1, w)
    box = (ys >= y1.view(n, 1, 1)) & (ys < y2.view(n, 1, 1)) & (xs >= x1.view(n, 1, 1)) & (xs < x2.view(n, 1, 1))
    mixed = torch.where(box.unsqueeze(1), b, a)
    lam_real = 1.0 - box.float().mean(dim=(1, 2))
    return mixed, lam_real


class RandAugmenter:
    """Per-image RandAugment on uint8 CPU tensors (each image gets its own random ops)."""

    def __init__(self, num_ops: int = 2, magnitude: int = 9):
        self.t = T.RandAugment(num_ops=num_ops, magnitude=magnitude)

    def batched(self, x: torch.Tensor, device: str = "cuda", group: int = 16) -> torch.Tensor:
        """Fast GPU version: images are augmented in small groups that share the same random ops."""
        xs = x.to(device)
        out = torch.empty_like(xs)
        for i in range(0, len(xs), group):
            out[i:i + group] = self.t(xs[i:i + group])
        return out.cpu()

    def __call__(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        out = x.clone()
        idxs = range(len(out)) if mask is None else torch.nonzero(mask).flatten().tolist()
        for i in idxs:
            out[i] = self.t(out[i])
        return out
