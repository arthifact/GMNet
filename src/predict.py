"""Step 4: macromodels for new recordings; no velocity model needed.

Input: raw shots [N, 5, 1000, 70] in OpenFWI format (five sources, 70 receivers every 10 m, 1 ms).
Output: gaussians.npz (the Gaussians [a, x, z, log σx, log σz, θ] and the background per model)
and velocity.npy, the drawn 70 x 70 macromodels in m/s.

    uv run src/predict.py              settings in config.toml [predict] -> outputs/<name>/predictions/

The shots are filtered to the band the model was trained on.
"""
from pathlib import Path

import numpy as np
import torch

from data import normalize
from gaussians import COLUMNS, render
from process import field_band
from train import device_for, load_model


@torch.no_grad()
def predict(run, shots, device='auto', batch=256):
    """Macromodels from the model of run for raw shots; the results go to run/predictions/."""
    device = device_for(device)
    model, saved = load_model(Path(run) / 'model/best.pt', device)
    shots, out = np.load(shots, mmap_mode='r'), Path(run) / 'predictions'
    if shots.ndim != 4 or shots.shape[1:] != (5, 1000, 70):
        raise ValueError(f'Expected raw shots [N,5,1000,70], found {shots.shape}.')
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for start in range(0, len(shots), batch):
        band = torch.from_numpy(field_band(shots[start:start + batch], *saved['band_hz']))   # the trained band
        outputs.append(model(normalize(band).to(device)))
    g, trend = (torch.cat([output[i] for output in outputs]).cpu().double() for i in (0, 1))
    np.savez_compressed(out / 'gaussians.npz', gaussians=g.numpy(), background=trend.numpy(), columns=COLUMNS)
    np.save(out / 'velocity.npy', render(g, trend).float().numpy())
    return g, trend


if __name__ == '__main__':
    import config
    c = config.load()
    g, _ = predict(c['run'], **c['predict'])
    print(f'{len(g)} macromodels written to {c["run"] / "predictions"}')
