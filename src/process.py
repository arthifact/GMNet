"""Step 1: OpenFWI shot/model pairs in data_raw/ -> data_processed/, once for every band.

Shots [N,5,1000,70] become [N,5,63,35], every 16 ms and 20 m (32x fewer numbers), stored as
float16. They keep 0-31 Hz, all that 16 ms samples can hold; training picks a band inside it
(band_pass). Velocity models are smoothed with a 50 m Gaussian (the macromodel target). Each
family is split by whole files (by default the last 10% for test and the 5% before for
validation); a family with a single 500-example file uses rows 0-400 / 400-452 / 452-500. The
manifest records the split and where every file came from.

    uv run src/process.py              settings in config.toml [data]
"""
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import gaussian_filter


FAMILIES = ['flatvel-a', 'flatvel-b', 'curvevel-a', 'curvevel-b', 'flatfault-a', 'flatfault-b',
            'curvefault-a', 'curvefault-b', 'style-a', 'style-b']
SPLITS = ('train', 'validation', 'test')
BAND_HZ = (4., 10.)          # the field band (Sun et al., GJI 2023): the realism check
DEFAULT_HZ = (0., 10.)      # to compare with other networks; above 10 Hz did not help
MAX_HZ = 31.                # 16 ms samples hold up to 31.25 Hz; the filter reaches 0 just below
SHOT_SHAPE = (5, 63, 35)
TARGET_SIGMA_M = 50.        # macromodel smoothing, whatever the band: about half a wavelength at 10 Hz
CHUNK = 20                  # shots per read
SINGLE_FILE = dict(train=(0, 400), validation=(400, 452), test=(452, 500))
PAIRS = (('data', 'model'), ('seis', 'vel'))


def low_pass(f, cutoff_hz):
    """1 below 0.7 cutoff, a cosine roll-off to 0 at the cutoff."""
    edge = .7 * cutoff_hz
    return np.where(f <= edge, 1., .5 * (1 + np.cos(np.pi * np.clip((f - edge) / (cutoff_hz - edge), 0, 1))))


def band_filter(f, low_hz, high_hz):
    """Zero-phase band-pass gain. For 4-10 Hz: 0 below 2.8 Hz, 1 from 4 to 7 Hz, 0 at 10 Hz.
    A low end of 0 keeps everything below the top."""
    if not 0 <= low_hz < high_hz <= MAX_HZ:
        raise ValueError(f'A band needs 0 <= low < high <= {MAX_HZ:g} Hz, got {low_hz:g}-{high_hz:g}.')
    return low_pass(f, high_hz) * (1 - low_pass(f, low_hz) if low_hz > 0 else 1.)


def downsample(shots):
    """Raw OpenFWI shots [N,5,1000,70] (1 ms, 10 m) -> 0-31 Hz [N,5,63,35] (16 ms, 20 m).

    Mean of 4 samples (4 ms, the first centred at 1.5 ms); the 0-31 Hz filter; every 4th sample;
    and the mean of each pair of receivers.
    """
    pooled = np.asarray(shots, np.float64).reshape(len(shots), 5, 250, 4, 70).mean(3)
    keep = band_filter(np.fft.rfftfreq(500, .004), 0., MAX_HZ)
    band = np.fft.irfft(np.fft.rfft(pooled, 500, axis=2) * keep[:, None], 500, axis=2)[:, :, :250:4]
    return .5 * (band[..., 0::2] + band[..., 1::2])


def band_pass(shots, low_hz, high_hz):
    """Processed shots [N,5,63,35] -> the band low_hz-high_hz. Matches filtering the raw shots
    directly to within 1% (the data hold nothing above 31 Hz, so filtering commutes with sampling)."""
    n = 2 * shots.shape[2]                                          # padding against wrap-around
    keep = band_filter(np.fft.rfftfreq(n, .016), low_hz, high_hz)
    spectrum = np.fft.rfft(np.asarray(shots, np.float64), n, axis=2)
    return np.fft.irfft(spectrum * keep[:, None], n, axis=2)[:, :, :shots.shape[2]]


def field_band(shots, low_hz=BAND_HZ[0], high_hz=BAND_HZ[1]):
    """Raw OpenFWI shots [N,5,1000,70] -> the band low_hz-high_hz [N,5,63,35], as training sees it."""
    return band_pass(downsample(shots), low_hz, high_hz)


def save_npy(path, array):
    pending = Path(path).with_suffix('.pending.npy')
    np.save(pending, array)
    os.replace(pending, path)


def smooth(models, sigma_m=TARGET_SIGMA_M, spacing_m=10.):
    """[N,1,70,70] velocities -> 50 m Gaussian-smoothed float32 (reflect boundary, 4σ kernel)."""
    out = np.empty(models.shape, np.float32)
    for i in range(len(models)):
        out[i, 0] = gaussian_filter(np.asarray(models[i, 0], np.float64), sigma_m / spacing_m,
                                    mode='reflect', truncate=4.)
    return out


def discover(root):
    """Shot/model file pairs per family, in OpenFWI family order and natural file order. A family
    folder is named as on OpenFWI's Drive (FlatVel_A) or as here (flatvel-a); its files may sit in
    subfolders, as on the Drive (FlatVel_A/data, FlatVel_A/model)."""
    key = lambda name: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)]
    folders = {}
    for p in Path(root).iterdir():
        if p.is_dir() and not p.name.startswith(('.', '_')):
            family = p.name.lower().replace('_', '-')
            if family in folders:
                raise ValueError(f'Two folders for {family}: {folders[family].name} and {p.name}.')
            folders[family] = p
    families = sorted(folders, key=lambda f: (FAMILIES.index(f) if f in FAMILIES else len(FAMILIES), f))
    pairs = []
    for family in families:
        files = {p.name: p for p in folders[family].rglob('*.npy') if not p.name.startswith('.')}
        for shot_prefix, model_prefix in PAIRS:
            for name in sorted((n for n in files if n.startswith(shot_prefix)), key=key):
                shots, model_name = files[name], model_prefix + name[len(shot_prefix):]
                models = files.get(model_name)
                if models is None:
                    raise FileNotFoundError(f'No model file {model_name} for {shots}.')
                pairs.append(dict(family=family, stem=shots.stem, shots=shots, models=models,
                                  count=int(np.load(shots, mmap_mode='r').shape[0])))
    if not pairs:
        raise ValueError(f'No OpenFWI shot/model pairs under {root}.')
    return pairs


def split(pairs, test=.1, validation=.05):
    """Add rows={split: [[start, end]]} to each pair; files never cross splits."""
    for family in dict.fromkeys(p['family'] for p in pairs):
        group = [p for p in pairs if p['family'] == family]
        if len(group) == 1:
            if group[0]['count'] != 500:
                raise ValueError(f'{family}: a single file must hold 500 examples to use the row split.')
            group[0]['rows'] = {s: [list(SINGLE_FILE[s])] for s in SPLITS}
            continue
        if len(group) < 3:
            raise ValueError(f'{family} needs one 500-example file or at least three files.')
        n_test, n_val = max(1, round(test * len(group))), max(1, round(validation * len(group)))
        for i, p in enumerate(group):
            name = 'test' if i >= len(group) - n_test else 'validation' if i >= len(group) - n_test - n_val else 'train'
            p['rows'] = {s: [[0, p['count']]] if s == name else [] for s in SPLITS}
    return pairs


def process(raw, out, test=.1, validation=.05):
    out = Path(out)
    if (out / 'manifest.json').exists():
        raise FileExistsError(f'{out} is already processed.')
    pairs = split(discover(raw), test, validation)
    files = []
    for p in pairs:
        shots = np.load(p['shots'], mmap_mode='r')
        models = np.load(p['models'], mmap_mode='r')
        n = p['count']
        if shots.shape[1:] != (5, 1000, 70):
            raise ValueError(f'{p["shots"]}: expected [N,5,1000,70], found {shots.shape}.')
        if models.shape != (n, 1, 70, 70):
            raise ValueError(f'{p["models"]}: expected [{n},1,70,70], found {models.shape}.')
        shot_path = Path('shots') / p['family'] / f'{p["stem"]}.npy'
        target_path = Path('targets') / p['family'] / f'{p["stem"]}.npy'
        for path in (shot_path, target_path):
            (out / path).parent.mkdir(parents=True, exist_ok=True)
        pending = out / shot_path.with_suffix('.pending.npy')
        low_file = np.lib.format.open_memmap(pending, mode='w+', dtype='float16', shape=(n, *SHOT_SHAPE))
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            low = downsample(shots[start:end])
            if not np.isfinite(low).all() or np.abs(low).max() >= 65504:
                raise ValueError(f'{p["shots"]}: nonfinite or outside the float16 range.')
            low_file[start:end] = low
        low_file.flush()
        del low_file
        pending.replace(out / shot_path)
        if not np.isfinite(models).all() or np.any(np.asarray(models) <= 0):
            raise ValueError(f'{p["models"]}: invalid velocities.')
        targets = smooth(np.asarray(models))[:, 0]
        save_npy(out / target_path, targets)
        files.append(dict(family=p['family'], stem=p['stem'], count=n, rows=p['rows'],
                          shots=str(shot_path), targets=str(target_path),
                          shots_source=str(p['shots']), models_source=str(p['models'])))
        print(f'processed {p["family"]}/{p["stem"]}: {n} examples', flush=True)
    manifest = dict(counts={s: sum(b - a for f in files for a, b in f['rows'][s]) for s in SPLITS},
                    shots=f'float16, 0-{MAX_HZ:g} Hz, every 16 ms from 1.5 ms, receiver pairs every 20 m from 5 m',
                    targets='float32, 50 m Gaussian smoothing, reflect boundary, 4-sigma truncation',
                    files=files)
    if not manifest['counts']['train']:
        raise ValueError('No training examples.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    import config
    c = config.load()['data']
    counts = process(c['raw'], c['processed'], c['test'], c['validation'])['counts']
    print(f'{counts} -> {c["processed"]}')
