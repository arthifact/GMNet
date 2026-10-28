"""The processed data (src/process.py), filtered to one band and held in memory.

No data loader: a whole split sits in memory as float16 (22 kB per example, so all ~470,000
OpenFWI Vel, Fault and Style models take about 10 GB) and each batch is one indexed copy to the GPU.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from process import band_pass


def normalize(shots):
    """Band shots [B,5,T,R] -> network input asinh(shots / their RMS). Each example is divided by its
    own RMS: the absolute amplitude of field recordings (source strength, coupling) is unknown."""
    shots = shots.float()
    return torch.asinh(shots / shots.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Split:
    """One split: shots [N,5,63,35] in the band (float16), targets [N,70,70] (m/s), and where each case came from."""
    def __init__(self, root, split, band):
        self.root = Path(root)
        manifest = json.loads((self.root / 'manifest.json').read_text())
        shots, targets, self.cases = [], [], []
        for record in manifest['files']:
            for start, end in record['rows'][split]:
                shots.append(np.load(self.root / record['shots'], mmap_mode='r')[start:end])
                targets.append(np.load(self.root / record['targets'], mmap_mode='r')[start:end])
                self.cases += [dict(family=record['family'], stem=record['stem'], row=row) for row in range(start, end)]
        if not self.cases:
            raise ValueError(f'Empty {split} split.')
        self.shots = torch.from_numpy(np.concatenate([band_pass(s, *band).astype(np.float16) for s in shots]))
        self.targets = torch.from_numpy(np.concatenate(targets).astype(np.float32))

    def __len__(self):
        return len(self.cases)

    def batch(self, index, device):
        """Network inputs (normalize) and targets for the cases `index`, on the device."""
        return normalize(self.shots[index].to(device)), self.targets[index].to(device)
