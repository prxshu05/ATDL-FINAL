"""Random noise images in [0,1] used as the starting point (no real data):
  whiteboard: GN = Gaussian, UN = uniform, Pink, PN = Perlin noise
  ZOPGA-KD paper (Appendix, Fig. 4): "smooth gradient noise" = constant colour gradient along a random direction
  DIP-KD (arXiv 2604.25794) image priors: "hier" = hierarchical multi-scale noise + rotation/elastic warp,
                                          "dip"  = two such images (or a plain colour) mixed through a smooth blob mask

Every function returns uint8 (n, 3, 32, 32). Each image gets its own random mean / contrast / colour so the
seed set is as diverse as possible ("root diversity").
"""

import torch
import torch.nn.functional as F


def _rand(n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, 1, 1, 1)


def _minmax(x):
    lo = x.amin((1, 2, 3), keepdim=True)
    hi = x.amax((1, 2, 3), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def _colourise(x):
    """x in [0,1]: random per-channel gain/offset so the noise is not grey."""
    n = len(x)
    gain = 0.4 + 0.6 * torch.rand(n, 3, 1, 1)
    off = (1 - gain) * torch.rand(n, 3, 1, 1)
    return x * gain + off


def _to_uint8(x):
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def gaussian(n):
    """Gaussian pixel noise N(mu, sigma), mu ~ U(0.2,0.8), sigma ~ U(0.05,0.35)."""
    return _to_uint8(_rand(n, 0.2, 0.8) + _rand(n, 0.05, 0.35) * torch.randn(n, 3, 32, 32))


def uniform(n):
    """Uniform pixel noise U(a, b) with random range."""
    a = _rand(n, 0.0, 0.5)
    b = a + _rand(n, 0.2, 0.5)
    return _to_uint8(a + (b - a) * torch.rand(n, 3, 32, 32))


def pink(n):
    """1/f^alpha noise (alpha ~ U(1,2): pink -> brown), i.e. natural-image-like power spectrum."""
    f = torch.fft.fftfreq(32)
    r = (f[:, None] ** 2 + f[None, :] ** 2).sqrt()
    r[0, 0] = 1.0
    alpha = _rand(n, 1.0, 2.0)
    spec = torch.fft.fft2(torch.randn(n, 3, 32, 32)) / r.view(1, 1, 32, 32) ** alpha
    return _to_uint8(_colourise(_minmax(torch.fft.ifft2(spec).real)))


def perlin(n, octaves: int = 4):
    """Perlin-style fractal noise: sum of smoothly interpolated random lattices (2x2 ... 16x16 grids)."""
    acc = torch.zeros(n, 3, 32, 32)
    persistence = _rand(n, 0.4, 0.7)
    for o in range(octaves):
        g = 2 ** (o + 1)
        acc += F.interpolate(torch.randn(n, 3, g, g), size=(32, 32), mode="bicubic", align_corners=False) * persistence ** o
    return _to_uint8(_colourise(_minmax(acc)))


def gradient(n):
    """Smooth gradient noise (ZOPGA-KD): colour c0 -> c1 along a random direction theta."""
    theta = 2 * torch.pi * torch.rand(n, 1, 1, 1)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing="ij")
    t = torch.cos(theta) * xx + torch.sin(theta) * yy                      # (n,1,32,32)
    t = (t - t.amin((2, 3), keepdim=True)) / (t.amax((2, 3), keepdim=True) - t.amin((2, 3), keepdim=True) + 1e-8)
    c0, c1 = torch.rand(n, 3, 1, 1), torch.rand(n, 3, 1, 1)
    return _to_uint8(c0 + (c1 - c0) * t)


# ---------------------------------------------------------------------------------------------------------------
# Rich "image priors" in the style of DIP-KD (Vo et al., arXiv 2604.25794): still pure noise, no data, no pretrained
# model -- but with multi-scale structure, geometric distortions and object-like blobs, so the teacher sees images
# that activate many more of its classes than flat pixel noise.
# ---------------------------------------------------------------------------------------------------------------
def _blur(x, sigma: float):
    """Separable Gaussian blur, reflect padding. x: (n, c, h, w) float."""
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def _hier_float(n, size: int):
    """Hierarchical noise (DIP-KD Eq. 3): sum_d softmax(a)_d * upsample(U[0,1]^{3 x 2^d x 2^d}),  d = 0..log2(size).
    Random softmax weights decide per image whether coarse blobs or fine texture dominate."""
    dmax = (size - 1).bit_length()
    w = torch.softmax(1.5 * torch.randn(n, dmax + 1), 1)
    acc = torch.zeros(n, 3, size, size)
    for d in range(dmax + 1):
        g = 2 ** d
        mode = "nearest" if torch.rand(1).item() < 0.3 else "bilinear"
        up = F.interpolate(torch.rand(n, 3, g, g), size=(size, size), mode=mode,
                           **({} if mode == "nearest" else {"align_corners": False}))
        acc += w[:, d].view(n, 1, 1, 1) * up
    return _minmax(acc)


def _nonlinear(x, out: int = 32):
    """Nonlinear transform (DIP-KD Eq. 4): rotation in [-45, 45] deg + elastic warp + random crop to out x out."""
    n, _, h, _ = x.shape
    th = (torch.rand(n) * 2 - 1) * torch.pi / 4
    cos, sin = torch.cos(th), torch.sin(th)
    A = torch.stack([torch.stack([cos, -sin, torch.zeros(n)], 1), torch.stack([sin, cos, torch.zeros(n)], 1)], 1)
    mid = out + 8
    grid = F.affine_grid(A * (mid / h), (n, 3, mid, mid), align_corners=False)
    disp = _blur(torch.randn(n, 2, mid, mid), sigma=4.0) * (0.5 + 1.5 * torch.rand(n, 1, 1, 1))   # elastic field
    grid = grid + disp.permute(0, 2, 3, 1)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    i, j = torch.randint(0, mid - out + 1, (2,)).tolist()
    return y[:, :, i:i + out, j:j + out]


def _blob_mask(n, size: int = 32, iters: int = 10):
    """Semantic cutmix mask (DIP-KD Eq. 5): uniform noise, repeatedly blurred and pushed to 0/1 -> smooth blobs."""
    m = torch.rand(n, 1, size, size)
    for _ in range(iters):
        m = _minmax(_blur(m, sigma=1.0 + torch.rand(1).item()))
        m = torch.where(m < 0.5, 2 * m ** 2, 1 - 2 * (1 - m) ** 2)
    return m


def hier(n):
    """Hierarchical multi-scale noise + rotation/elastic/crop, randomly colourised."""
    x = _nonlinear(_hier_float(n, 48))
    return _to_uint8(torch.where(torch.rand(n, 1, 1, 1) < 0.5, _colourise(x), x))


def dip(n, mono_p: float = 0.3):
    """DIP-KD style prior: two warped hierarchical-noise images (or a plain colour, prob. mono_p) combined
    through a smooth blob mask -> object-like regions on a background."""
    a = _nonlinear(_hier_float(n, 48))
    b = _nonlinear(_hier_float(n, 48))
    mono = torch.rand(n, 3, 1, 1).expand(n, 3, 32, 32)
    b = torch.where(torch.rand(n, 1, 1, 1) < mono_p, mono, b)
    a = torch.where(torch.rand(n, 1, 1, 1) < 0.5, _colourise(a), a)
    m = _blob_mask(n)
    return _to_uint8(m * a + (1 - m) * b)


KINDS = {"gaussian": gaussian, "uniform": uniform, "pink": pink, "perlin": perlin, "gradient": gradient,
         "hier": hier, "dip": dip}


def make_noise(n: int, kinds: str = "gaussian,uniform,pink,perlin,gradient", device: str = None) -> torch.Tensor:
    """n images split equally between the requested noise kinds, shuffled. device="cuda" builds them on the GPU
    (same code, much faster); the result is always returned on the CPU."""
    if device is not None and device != "cpu":
        with torch.device(device):
            return make_noise(n, kinds).cpu()
    names = [k.strip() for k in kinds.split(",") if k.strip()]
    for k in names:
        if k not in KINDS:
            raise ValueError(f"unknown noise kind '{k}', choose from {list(KINDS)}")
    sizes = [n // len(names) + (1 if i < n % len(names) else 0) for i in range(len(names))]
    x = torch.cat([KINDS[k](s) for k, s in zip(names, sizes) if s > 0])
    return x[torch.randperm(len(x))]
