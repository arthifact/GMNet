"""Step 3: errors of a trained model on one split, per case and per family, and its Gaussians.

geometry bound: the predicted centres and shapes with the 34 linear values (32 amplitudes,
    2 background terms) solved exactly against the truth: how good the geometry is alone.
FWI-ready: the field criterion for a starting model. First-arrival times through the predicted
    macromodel must stay within half a period of those through the true (unsmoothed) model at
    4 Hz, where field FWI starts: 125 ms, or waveform inversion cycle-skips. Needs the raw models.

    uv run src/evaluate.py             the model of config.toml on the test split -> outputs/<name>/evaluation/

The data are filtered to the band the model was trained on.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from data import Split, file_hash
from gaussians import COLUMNS, render, solve
from train import device_for, load_model
from traveltime import first_arrivals

FWI_START_HZ = 4.     # field FWI starts at 4 Hz, whatever band the network reads
HALF_PERIOD_MS = 500 / FWI_START_HZ


@torch.no_grad()
def evaluate(run, processed, split='test', device='auto', batch=64):
    """Score the model of run on one split; the results go to run/evaluation/."""
    device = device_for(device)
    model, saved = load_model(Path(run) / 'model/best.pt', device)
    if file_hash(Path(processed) / 'manifest.json') != saved['manifest_sha256']:
        raise ValueError('This is not the processed data the model was trained on.')
    data, out = Split(processed, split, saved['band_hz']), Path(run) / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    outputs = [model(data.batch(index, device)[0]) for index in torch.arange(len(data)).split(batch)]
    g, trend = (torch.cat([output[i] for output in outputs]).cpu().double() for i in (0, 1))
    truth, predicted = data.targets.double(), render(g, trend)
    error = lambda field: (field - truth).square().mean((1, 2)).sqrt()
    bound = torch.cat([render(*solve(g[i], truth[i])) for i in torch.arange(len(g)).split(batch)])
    cases = [dict(case, rmse_m_s=float(a), geometry_bound_m_s=float(b))
             for case, a, b in zip(data.cases, error(predicted), error(bound))]
    manifest = json.loads((Path(processed) / 'manifest.json').read_text())
    raw = {(f['family'], f['stem']): f['models_source'] for f in manifest['files']}
    if all(Path(path).exists() for path in raw.values()):
        for case, field in zip(cases, predicted.numpy()):
            true_model = np.load(raw[case['family'], case['stem']], mmap_mode='r')[case['row'], 0]
            miss = 1000 * np.abs(first_arrivals(field) - first_arrivals(true_model)).max()
            case.update(first_arrival_error_ms=float(miss), fwi_ready=float(miss < HALF_PERIOD_MS))
    else:
        print('FWI-ready skipped: the raw velocity models are no longer where process.py read them.')
    keys = [k for k in ('rmse_m_s', 'geometry_bound_m_s', 'first_arrival_error_ms', 'fwi_ready') if k in cases[0]]
    mean = lambda rows: {key: float(np.mean([r[key] for r in rows])) for key in keys}
    families = sorted({r['family'] for r in cases})
    summary = dict(split=split, count=len(cases), overall=mean(cases), half_period_ms=HALF_PERIOD_MS,
                   per_family={f: mean([r for r in cases if r['family'] == f]) for f in families})
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'per_case.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    np.savez_compressed(out / 'gaussians.npz', gaussians=g.numpy(), background=trend.numpy(), columns=COLUMNS)
    return summary


if __name__ == '__main__':
    import config
    c = config.load()
    print(json.dumps(evaluate(c['run'], c['data']['processed']), indent=2))
