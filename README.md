# GMNet: Gaussian Macromodel Network

GMNet predicts a seismic velocity macromodel from the low frequencies of five OpenFWI shot gathers:
0-10 Hz, or the 4-10 Hz that field surveys record.
A macromodel is the smooth starting model for imaging and full-waveform inversion.
GMNet writes it as 32 anisotropic Gaussians plus a depth-linear background,
194 total parameters, and predicts them in one pass.

## Architecture
```
shots [5,63,35] ─CNN─> 8×5 map ─resize─> 4×8 cells ─transformer─> 32 Gaussians + background ─render─> 70×70 model
```

- **Input.** Five shots every 16 ms and 20 m, band-passed to 0-10 Hz or another band such as
  4-10 Hz. Each example is divided by its own RMS and passed through `asinh`, so the absolute
  amplitude of the survey does not matter.

- **CNN.** Eight 3 x 3 convolution blocks (GroupNorm, LeakyReLU), three with stride 2, 32 to 128 channels.

- **One cell per Gaussian.** The last map is resized to a 4 x 8 lattice. Cell k sits over Gaussian
  k's starting place: later times over deeper Gaussians, receiver columns over x.

- **Mixing.** The 32 cells plus one background token pass two transformer layers, so every Gaussian
  sees the others.

- **Heads.** Two MLPs shared by all cells give each Gaussian its shape [x, z, σx, σz, θ], as an
  offset from the lattice, and its velocity. A third gives the background. All start at zero, so
  training starts from the lattice at 3000 m/s.

- **Output.** The Gaussians are drawn in closed form. Centres stay in the model, widths in
  50-300 m and the aspect ratio ≤ 3, so every prediction is a smooth, valid model.

## Training

- The loss is the error of the drawn model; nothing about the Gaussians is supervised.
- Half of each batch is mirrored left-right.
- AdamW, 100 epochs, batch 64. In total 0.90M weights and 0.22 GFLOPs per example.

Why each choice, and the measurement behind it: [docs/DESIGN.md](docs/DESIGN.md).

## Use

Install [uv](https://docs.astral.sh/uv/), then download OpenFWI family folders from their Google
Drive and extract them into `data_raw/`, e.g. `data_raw/FlatVel_A/` (links in
[data_raw/README.md](data_raw/README.md)).

```bash
uv sync
```

Every setting lives in [config.toml](config.toml), and every command reads it. Results go to
`outputs/<name>/` ([outputs/README.md](outputs/README.md)).

1. **Process**, once: keep 0-31 Hz on a 16 ms, 20 m grid, smooth the velocity models into
   macromodel targets, and split each family into train, validation and test by file.
   `data_raw/` is only read; the result goes to `data_processed/`, about 30x smaller.

```bash
uv run src/process.py
```

2. **Train** on the band in `[train]`: 0-10 Hz to compare with other networks, or 4-10 Hz, the band
   field surveys record ([why these bands](docs/DESIGN.md#3-which-frequencies)). It picks CUDA,
   then MPS, then CPU.

```bash
uv run src/train.py
```

3. **Evaluate** that model on the test split: RMSE, geometry bound and FWI-ready.

```bash
uv run src/evaluate.py
```

4. **Predict** macromodels for the raw shots `[N, 5, 1000, 70]` named in `[predict]`.

```bash
uv run src/predict.py
```

**The realism check:** set `name = "4-10Hz"` and `band = [4, 10]` in `config.toml`, then train,
evaluate and predict again. Each name keeps its own folder in `outputs/`.

**Tests:**

```bash
uv run tests/test_steps.py
```

## License

[MIT](LICENSE)
