"""GMNet: one frequency band of five shot gathers -> K Gaussians + a depth-linear background.

    shots [5, 63, 35] ─CNN─> map ─resize─> one cell per Gaussian ─transformer─> 6K + 2 numbers

The last map of a small CNN IS the Gaussians: resized to the nx x nz lattice (4 x 8 = 32 by
default), cell k sits over Gaussian k's starting place (time row ~ depth row, receiver column ~ x).
Two transformer layers let the Gaussians see each other, and two MLPs shared by all cells turn each
cell into its offsets from the start and its velocity. One extra token, the mean of the cells, gives the background. One pass, no
decoder: the numbers draw the 70 x 70 model in closed form (gaussians.render). Each cell already
knows which Gaussian it is from where it sits, so no slot embeddings are needed.
"""
import torch
from torch import nn
from torch.nn import functional as F

from gaussians import gaussians, initial

ARCHITECTURE = 'gmnet_v1'


def block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a, b, 3, stride, 1), nn.GroupNorm(8, b), nn.LeakyReLU(.2))


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, n_out))


class GMNet(nn.Module):
    def __init__(self, lattice=(4, 8), channels=128, layers=2):
        super().__init__()
        self.config = dict(lattice=list(lattice), channels=channels, layers=layers)
        (nx, nz), c = lattice, channels
        self.lattice = (nz, nx)
        self.cnn = nn.Sequential(block(5, c // 4), block(c // 4, c // 4),            # 63 x 35
                                 block(c // 4, c // 2, 2), block(c // 2, c // 2),    # 32 x 18
                                 block(c // 2, c, 2), block(c, c),                   # 16 x 9
                                 block(c, c, 2), block(c, c))                        # 8 x 5
        layer = nn.TransformerEncoderLayer(c, 4, 2 * c, 0., batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(c), enable_nested_tensor=False)
        self.move, self.velocity, self.trend = mlp(c, 5), mlp(c, 1), mlp(c, 2)
        self.register_buffer('start', initial(lattice))
        for head in (self.move, self.velocity, self.trend):      # start exactly at the lattice, 3000 m/s
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, shots):
        """Band shots [B,5,63,35] (data.normalize) -> Gaussians [B,K,6] and background [B,2]."""
        cells = F.interpolate(self.cnn(shots), size=self.lattice, mode='bilinear', align_corners=True)
        h = cells.flatten(2).transpose(1, 2)                     # slot order: depth row, then x
        h = self.mix(torch.cat((h, h.mean(1, keepdim=True)), 1))
        raw = (self.start + self.move(h[:, :-1])).clamp(-8, 8)
        amplitude = 1500 * self.velocity(h[:, :-1])[..., 0]
        return gaussians(amplitude, raw), self.trend(h[:, -1])
