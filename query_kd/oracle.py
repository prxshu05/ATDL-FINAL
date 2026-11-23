"""Black-box teacher with a query counter. Every image sent to `query` costs one unit of budget."""

import torch
import torch.nn.functional as F

from .data import normalize


def amp_ctx(device):
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict_logprobs(model, x_uint8: torch.Tensor, device: str, batch_size: int = 1000) -> torch.Tensor:
    """Free forward pass (used for the student, and for evaluation only). Returns CPU float32 log-probs."""
    model.eval()
    out = []
    for i in range(0, len(x_uint8), batch_size):
        xb = normalize(x_uint8[i:i + batch_size].to(device, non_blocking=True))
        with amp_ctx(device):
            z = model(xb)
        out.append(F.log_softmax(z.float(), dim=1).cpu())
    return torch.cat(out) if out else torch.empty(0, 100)


class TeacherOracle:
    """Wraps the teacher as a paid black-box API.

    The API returns the full probability vector (stored as log-probabilities, which equal the teacher
    logits up to a per-sample constant, so the softmax temperature can still be changed later).
    In hard-label experiments the training code only uses argmax of it.
    """

    def __init__(self, model, device: str, budget: int):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.budget = budget
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def query(self, x_uint8: torch.Tensor) -> torch.Tensor:
        n = len(x_uint8)
        if self.used + n > self.budget:
            raise RuntimeError(f"Query budget exceeded: used {self.used} + {n} > {self.budget}")
        self.used += n
        return predict_logprobs(self.model, x_uint8, self.device)
