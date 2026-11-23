"""PGA on the STUDENT: Projected Gradient Ascent in input space to synthesise class-targeted, diverse images.

Follows the PGA of ZOPGA-KD (Mukherjee et al., arXiv:2611.00908, Eq. 3-5, Alg. 1) with two changes:
  1. the ascent is done on the STUDENT S (our own model -> exact gradients by back-prop, free, no teacher queries)
     instead of zeroth-order gradient estimates on the teacher (which costs millions of teacher queries);
  2. a KL DIVERSITY term is added so images of the same class do not collapse to the same point.
The teacher is queried only ONCE per finished image (to label it) -> the method stays black-box w.r.t. the teacher.

  s(x) = softmax(S(x)),  r(x) = softmax over the K-1 non-target classes (what x looks like *besides* class c)

  J(x) = ln s_c(x)                                                          reach target class c
       + lam_intra * mean_{j: c_j = c, j != i} SKL( r(x_i), r(x_j) )         diversity: new vs new, same class
       + lam_old   * min_{o in D, class c} KL( s(x_i) || s(x_o) )             diversity: new vs old data  (KL(n,o) up)

  g = grad_x J,   m <- beta*m + (1-beta)*g,   x <- clip_[0,1]( x + alpha_t * m / ||m||_2 )        (Eq. 4-5)
  alpha_t: cosine from alpha_max to alpha_min.  An image retires when s_c(x) >= tau (target confidence, e.g. 0.8/0.9).
  If s_c(x) < 0.3 after `stall_steps`, the image is restarted from fresh noise (as in ZOPGA-KD).
  KL terms are capped at `kl_cap`.  Optional random jitter (shift+flip) makes images less student-specific.
  grad_res=(8, 16): the gradient is restricted to a low-resolution subspace (8x8 below 0.5 confidence, 16x16 above,
  upsampled), like the low-res perturbation directions of ZOPGA-KD -> smooth images instead of checkerboards.
"""

import math

import torch
import torch.nn.functional as F

from .data import normalize
from .oracle import amp_ctx


def _jitter(x, max_shift: int = 2):
    dx, dy = torch.randint(-max_shift, max_shift + 1, (2,)).tolist()
    x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
    return torch.flip(x, dims=(3,)) if torch.rand(1).item() < 0.5 else x


class PGAGenerator:
    def __init__(self, steps: int = 300, alpha_max: float = 0.5, alpha_min: float = 0.1, beta: float = 0.9,
                 conf=(0.8, 0.9), lam_intra: float = 0.1, lam_old: float = 0.1, kl_cap: float = 5.0,
                 jitter: bool = True, stall_steps: int = 150, batch_size: int = 500, num_ref: int = 5000,
                 grad_res=(8, 16)):
        self.steps, self.alpha_max, self.alpha_min, self.beta = steps, alpha_max, alpha_min, beta
        self.conf = tuple(conf)
        self.lam_intra, self.lam_old, self.kl_cap = lam_intra, lam_old, kl_cap
        self.jitter, self.stall_steps, self.bs, self.num_ref = jitter, stall_steps, batch_size, num_ref
        self.grad_res = tuple(grad_res) if grad_res else None

    def _smooth(self, g, conf):
        """Keep only the low-resolution part of the gradient (ZOPGA-KD perturbation space, adaptive resolution):
        S1 x S1 while the target confidence < 0.5, S2 x S2 afterwards, bilinearly upsampled to 32 x 32.
        Without this the ascent produces high-frequency checkerboard patterns that do not resemble real images."""
        h, w = g.shape[-2:]
        lo = F.interpolate(F.adaptive_avg_pool2d(g, self.grad_res[0]), size=(h, w), mode="bilinear", align_corners=False)
        hi = F.interpolate(F.adaptive_avg_pool2d(g, self.grad_res[1]), size=(h, w), mode="bilinear", align_corners=False)
        return torch.where((conf < 0.5).view(-1, 1, 1, 1), lo, hi)

    def _alpha(self, t):
        return self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1 + math.cos(math.pi * t / max(1, self.steps)))

    def generate(self, student, x0_uint8, targets, device, noise_fn=None, ref_lp=None, ref_y=None, log=print):
        """x0_uint8: (N,3,32,32) noise init; targets: (N,) class ids; noise_fn(n) -> fresh noise for restarts;
        ref_lp / ref_y: student log-probs and (teacher) labels of the data already in D, for the KL(new||old) term.
        Returns uint8 images (CPU) and a stats dict.  Student forward/backward passes are counted in stats."""
        student.eval()
        trainable = [p.requires_grad for p in student.parameters()]
        for p in student.parameters():
            p.requires_grad_(False)
        use_old = ref_lp is not None and self.lam_old > 0 and len(ref_lp) > 0
        if use_old:
            if len(ref_lp) > self.num_ref:
                sel = torch.randperm(len(ref_lp))[: self.num_ref]
                ref_lp, ref_y = ref_lp[sel], ref_y[sel]
            ref_lp, ref_y = ref_lp.to(device).float(), ref_y.to(device)
        out = []
        tot = {"conf": 0.0, "retired": 0.0, "restarts": 0, "intra": 0.0, "old": 0.0, "student_passes": 0}
        for ci in range(math.ceil(len(x0_uint8) / self.bs)):
            x = x0_uint8[ci * self.bs:(ci + 1) * self.bs].to(device).float() / 255.0
            y = targets[ci * self.bs:(ci + 1) * self.bs].to(device)
            b = len(x)
            tau = torch.tensor(self.conf, device=device)[torch.randint(len(self.conf), (b,), device=device)]
            same = (y[:, None] == y[None, :]) & ~torch.eye(b, dtype=torch.bool, device=device)
            if use_old:
                cls_mask = ref_y[None, :] == y[:, None]
                has_cls = cls_mask.any(1)
            m = torch.zeros_like(x)
            active = torch.ones(b, dtype=torch.bool, device=device)
            restarted = torch.zeros(b, dtype=torch.bool, device=device)
            d_intra = d_old = torch.zeros((), device=device)
            for t in range(self.steps):
                if not active.any():
                    break
                x.requires_grad_(True)
                with amp_ctx(device):
                    logits = student(normalize(_jitter(x) if self.jitter else x))
                logits = logits.float()
                lp = F.log_softmax(logits, 1)
                lpc = lp.gather(1, y[:, None]).squeeze(1)
                J = lpc.sum()
                if self.lam_intra > 0 and same.any():
                    r = F.log_softmax(logits.scatter(1, y[:, None], -30.0), 1)
                    pr = r.exp()
                    kl = (pr * r).sum(1, keepdim=True) - pr @ r.t()          # KL(r_i || r_j)
                    skl = (0.5 * (kl + kl.t())).clamp(max=self.kl_cap)
                    per = (skl * same).sum(1) / same.sum(1).clamp_min(1)
                    d_intra = per.mean()
                    J = J + self.lam_intra * per.sum()
                if use_old:
                    p = lp.exp()
                    kl_no = (p * lp).sum(1, keepdim=True) - p @ ref_lp.t()   # KL(s_new || s_old)
                    near = torch.where(has_cls, kl_no.masked_fill(~cls_mask, float("inf")).min(1).values,
                                       kl_no.min(1).values).clamp(max=self.kl_cap)
                    d_old = near.mean()
                    J = J + self.lam_old * near.sum()
                g, = torch.autograd.grad(J, x)
                tot["student_passes"] += b
                x = x.detach()
                with torch.no_grad():
                    conf = lpc.detach().exp()
                    active &= conf < tau                                     # retire images that reached tau
                    if self.grad_res:
                        g = self._smooth(g, conf)
                    m = self.beta * m + (1 - self.beta) * g
                    step = m / m.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
                    x = torch.where(active.view(-1, 1, 1, 1), (x + self._alpha(t) * step).clamp(0, 1), x)
                    if noise_fn is not None and t == self.stall_steps:      # restart stalled images once
                        stall = active & (conf < 0.3)
                        if stall.any():
                            k = int(stall.sum())
                            x[stall] = noise_fn(k).to(device).float() / 255.0
                            m[stall] = 0
                            restarted |= stall
                            tot["restarts"] += k
            with torch.no_grad(), amp_ctx(device):
                pc = F.softmax(student(normalize(x)).float(), 1).gather(1, y[:, None]).squeeze(1)
            tot["conf"] += pc.sum().item()
            tot["retired"] += (pc >= tau - 1e-3).sum().item()
            tot["intra"] += d_intra.detach().item() * b
            tot["old"] += d_old.detach().item() * b
            out.append((x * 255).round().clamp(0, 255).to(torch.uint8).cpu())
        for p, tr in zip(student.parameters(), trainable):
            p.requires_grad_(tr)
        n = len(x0_uint8)
        st = {"conf": tot["conf"] / n, "reached_tau": tot["retired"] / n, "restarts": tot["restarts"],
              "intra_skl": tot["intra"] / n, "kl_new_old": tot["old"] / n, "student_passes": tot["student_passes"]}
        log(f"    PGA on student: {n} images | mean student conf on target {st['conf']:.3f} | reached tau "
            f"{100 * st['reached_tau']:.1f}% | restarts {st['restarts']} | intra-class SKL {st['intra_skl']:.3f} | "
            f"KL(new||old) {st['kl_new_old']:.3f}")
        return torch.cat(out), st
