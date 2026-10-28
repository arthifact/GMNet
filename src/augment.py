"""Left-right mirror, the one augmentation, run on the GPU with the batch.

OpenFWI's five sources sit at 0, 170, 340, 520 and 690 m (checked on flat models). Mirroring a
model about its centre maps source k to source 4-k and each receiver pair to its mirror pair,
so flipping both axes is exact for sources 0, 1, 3 and 4. The middle source mirrors to 350 m,
10 m (half a receiver pair) from where it was recorded.
"""
import torch


def mirror_shots(shots):
    """Shots [B, 5 sources, time, receivers] -> shots of the mirrored model."""
    return shots.flip(1).flip(-1)


def mirror(shots, fields, flip):
    """Mirror shots [B,5,T,R] and models [B,70,70] where `flip` [B] is true."""
    return (torch.where(flip[:, None, None, None], mirror_shots(shots), shots),
            torch.where(flip[:, None, None], fields.flip(-1), fields))
