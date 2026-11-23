"""CIFAR-style ResNets (3x3 stem, no max-pool).

State-dict compatible with the original project's checkpoints
(e.g. experiments/checkpoints/teacher_best.pth), so an existing teacher can be reused.
"""

import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

_ARCHS = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
}


def build_model(arch: str, num_classes: int = 100) -> nn.Module:
    arch = arch.lower().replace("-", "").replace("_", "")
    if arch not in _ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_ARCHS)}")
    block, layers = _ARCHS[arch]
    model = ResNet(block, layers, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def load_weights(model: nn.Module, path: str, device="cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return ckpt if isinstance(ckpt, dict) else {}


def save_weights(model: nn.Module, path: str, **extra):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)
