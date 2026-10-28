"""The macromodel representation: 3000 m/s + a depth-linear background + K Gaussians.

v(x, z) = 3000 + 1000 (b0 + b1 (z/690 - 1/2)) + Σ_k a_k exp(-½ (r - μ_k)ᵀ Σ_k⁻¹ (r - μ_k))

A Gaussian is [a, x, z, log σx, log σz, θ]: amplitude (m/s), centre (m), widths (m), angle.
K = 32 by default (a 4 x 8 lattice), so with the two background terms 32 x 6 + 2 = 194 numbers.
"""
import math

import numpy as np
import torch

SIZE, EXTENT = 70, 690.                            # a 70 x 70 grid at 10 m
LOG_MIN, LOG_MAX = math.log(50.), math.log(300.)  # widths σ: 50 m (the target's smoothing) to 300 m
                                                   # (about 700 m across, a 4 Hz wavelength at 3000 m/s)
LOG_ASPECT = math.log(3.)                          # long / short width <= 3
COLUMNS = ['amplitude_m_s', 'x_m', 'z_m', 'log_sigma_x_m', 'log_sigma_z_m', 'theta_rad']


def place(raw):
    """Unconstrained [..., 5] -> geometry [x, z, log σx, log σz, θ] inside the bounds."""
    half = .5 * LOG_ASPECT * raw[..., 3].tanh()        # half the log aspect ratio
    edge = (half.square() + 1e-12).sqrt()
    mean = LOG_MIN + edge + (LOG_MAX - LOG_MIN - 2 * edge) * raw[..., 2].sigmoid()
    return torch.stack((EXTENT * raw[..., 0].sigmoid(), EXTENT * raw[..., 1].sigmoid(),
                        mean + half, mean - half, math.pi / 2 * raw[..., 4]), -1)


def gaussians(amplitude, raw):
    """Amplitudes [B,K] (m/s) and raw geometry [B,K,5] -> Gaussians [B,K,6]."""
    return torch.cat((amplitude[..., None], place(raw)), -1)


def grid(size, like):
    """Sample positions in metres: 0, 10, ..., 690 for the full grid."""
    if size == SIZE:
        return torch.arange(SIZE, device=like.device, dtype=like.dtype) * 10
    return torch.linspace(0, EXTENT, size, device=like.device, dtype=like.dtype)


def kernels(g, size=SIZE):
    """Unit-height Gaussians [B,K,size,size] on the grid: exp(-½ dᵀ Σ⁻¹ d), d = (x - μx, z - μz).

    Σ⁻¹ is built from the angle and widths on the small [B,K] tensors, so the full grid sees only
    one outer product, two additions and the exponential.
    """
    r = grid(size, g)
    dx, dz = r - g[..., 1, None], r - g[..., 2, None]                    # [B,K,size] along x, along z
    c, s = g[..., 5].cos(), g[..., 5].sin()
    px, pz = (-2 * g[..., 3]).exp(), (-2 * g[..., 4]).exp()               # 1 / σ²
    xx, zz, xz = c * c * px + s * s * pz, s * s * px + c * c * pz, c * s * (px - pz)
    return (-.5 * (xx[..., None] * dx.square())[..., None, :] - .5 * (zz[..., None] * dz.square())[..., :, None]
            - (xz[..., None] * dz)[..., :, None] * dx[..., None, :]).exp()


def background(size, like):
    """[2,size,size]: the constant and linear-in-depth background terms (km/s)."""
    depth = (torch.arange(SIZE, device=like.device, dtype=like.dtype) / (SIZE - 1) - .5 if size == SIZE
             else torch.linspace(-.5, .5, size, device=like.device, dtype=like.dtype))
    ones = torch.ones(size, size, device=like.device, dtype=like.dtype)
    return torch.stack((ones, depth[:, None].expand(size, size)))


def render(g, trend, size=SIZE):
    """Gaussians [B,K,6] and background [B,2] -> velocity [B,size,size] in m/s."""
    return 3000 + torch.einsum('bk,bkzx->bzx', g[..., 0], kernels(g, size)) \
        + 1000 * torch.einsum('bt,tzx->bzx', trend, background(size, trend))


def solve(g, field, ridge=1e-4):
    """The field is linear in amplitudes and background: their least-squares values for this geometry."""
    size, count = field.shape[-1], g.shape[1]
    k = kernels(g, size).flatten(2)
    columns = torch.cat((k, background(size, field).flatten(1)[None].expand(len(g), -1, -1)), 1)
    gram = columns @ columns.mT
    eye = torch.eye(count + 2, dtype=gram.dtype, device=gram.device)
    gram = gram + ridge * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye
    x = torch.linalg.solve(gram, columns @ ((field - 3000) / 1000).flatten(1)[..., None])[..., 0]
    return torch.cat((1000 * x[:, :count, None], g[..., 1:]), -1), x[:, count:]


def initial(lattice=(4, 8)):
    """Start geometry [K,5]: nx columns by nz depth rows of Gaussians, as wide as their spacing
    allows (150 x 65 m for 4 x 8). Amplitudes start at zero: the start is 3000 m/s everywhere."""
    nx, nz = lattice
    sx = min(max(600 / nx, 55.), 285.)
    sz = min(max(520 / nz, 55., sx / 2.9), 285.)                    # inside the bounds, aspect ≤ 3
    logit = lambda p: math.log(p / (1 - p))
    half = .5 * math.log(sx / sz)
    edge = math.sqrt(half ** 2 + 1e-12)
    fraction = (.5 * math.log(sx * sz) - LOG_MIN - edge) / (LOG_MAX - LOG_MIN - 2 * edge)
    raw = torch.tensor([[logit(x), logit(z), logit(fraction), math.atanh(half / (.5 * LOG_ASPECT)), 0.]
                        for z in np.linspace(.02, .98, nz) for x in np.linspace(.02, .98, nx)])
    return raw.float()
