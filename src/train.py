"""Step 2: train GMNet on one band of the processed data.

The only loss is the error of the drawn macromodel; nothing about the Gaussians is supervised.
Half of the examples are mirrored left-right. A run is never overwritten.

    uv run src/train.py                settings in config.toml [train] -> outputs/<name>/model/
"""
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from augment import mirror
from data import Split, file_hash
from gaussians import render
from model import ARCHITECTURE, GMNet
from process import DEFAULT_HZ

LOSS_GRID = 35         # the loss draws the model on 35 x 35 (field_loss)
LATTICE = (4, 8)       # 32 Gaussians, where the error levels off (docs/DESIGN.md)
WEIGHT_DECAY, WARMUP = 1e-4, 3


def device_for(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return name


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != ARCHITECTURE:
        raise ValueError(f'Expected {ARCHITECTURE}, got {checkpoint["architecture"]}.')
    model = GMNet(**checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


def field_loss(g, trend, target):
    """Squared error of the drawn field in (km/s)².

    The Gaussians are drawn on a 35 x 35 grid (20 m): they are at least 50 m wide, so this error
    matches the 70 x 70 one (168.1 vs 167.8 m/s on a test set) at a quarter of the cost.
    """
    target = F.interpolate(target[:, None], size=(LOSS_GRID, LOSS_GRID), mode='bilinear', align_corners=True)[:, 0]
    return ((render(g, trend, LOSS_GRID) - target) / 1000).square().mean()


@torch.no_grad()
def rmse(model, data, device, cases=None, batch=128):
    """Mean per-model RMSE (m/s) of the final field on the full 70 x 70 grid."""
    model.eval()
    total = 0.
    for index in (torch.arange(len(data)) if cases is None else cases).split(batch):
        shots, target = data.batch(index, device)
        total += (render(*model(shots)) - target).square().mean((1, 2)).sqrt().sum()
    return float(total) / (len(data) if cases is None else len(cases))


def schedule(base, epoch, epochs, warmup):
    """Linear warmup from 10%, then cosine decay to 5% of the base rate."""
    if epoch <= warmup:
        return base * (.1 + .9 * epoch / warmup)
    return base * (.05 + .95 * .5 * (1 + math.cos(math.pi * (epoch - warmup - 1) / max(1, epochs - warmup - 1))))


def train(run, processed, band=DEFAULT_HZ, epochs=100, batch_size=64, learning_rate=1e-3, seed=0,
          device='auto', config_text=None):
    """Train on the band of the processed data; the model and its history go to run/model/."""
    torch.manual_seed(seed)
    device = device_for(device)
    data, validation = Split(processed, 'train', band), Split(processed, 'validation', band)
    model = GMNet(LATTICE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=WEIGHT_DECAY, fused=device != 'cpu')
    out = Path(run) / 'model'
    out.mkdir(parents=True)                                   # never overwrites a run
    if config_text:
        (Path(run) / 'config.toml').write_text(config_text)
    print(f'{len(data)} training examples, {band[0]:g}-{band[1]:g} Hz, {device}', flush=True)
    probe = torch.arange(0, len(data), 8)                     # a fixed eighth of training: under- or over-fitting?
    generator = torch.Generator().manual_seed(seed)
    best, history, clock = float('inf'), [], 0.
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        for group in optimizer.param_groups:
            group['lr'] = schedule(learning_rate, epoch, epochs, WARMUP)
        model.train()
        total = 0.
        for index in torch.randperm(len(data), generator=generator).split(batch_size):
            shots, target = data.batch(index, device)
            shots, target = mirror(shots, target, (torch.rand(len(index), generator=generator) < .5).to(device))
            loss = field_loss(*model(shots), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total = total + loss.detach() * len(index)
        total = float(total) / len(data)                      # waits for the GPU: the clock is training time
        clock += time.monotonic() - started
        score, fit = rmse(model, validation, device), rmse(model, data, device, probe)
        history.append(dict(epoch=epoch, loss=total, validation_rmse_m_s=score, train_probe_rmse_m_s=fit,
                            training_seconds=clock, learning_rate=optimizer.param_groups[0]['lr']))
        (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        if score < best:
            best = score
            torch.save(dict(architecture=ARCHITECTURE, config=model.config, band_hz=[float(f) for f in band],
                            manifest_sha256=file_hash(Path(processed) / 'manifest.json'),
                            model=model.state_dict(), epoch=epoch, validation_rmse_m_s=score), out / 'best.pt')
        print(f'epoch {epoch}: validation {score:.1f} m/s, train probe {fit:.1f}, best {best:.1f}, '
              f'{clock / 60:.1f} min of training', flush=True)
    return history


if __name__ == '__main__':
    import config
    c = config.load()
    train(c['run'], c['data']['processed'], **c['train'], config_text=c['path'].read_text())
