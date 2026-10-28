# GMNet: Gaussian Macromodel Network

GMNet predicts a seismic velocity macromodel, the smooth starting model for full-waveform
inversion, as 32 Gaussians from the low frequencies of five OpenFWI shot gathers, in one pass.
How it works and why: [docs/DESIGN.md](docs/DESIGN.md).

<img src="docs/figures/shots_to_gaussians.png" width="640" alt="A shot gather, the true macromodel and GMNet's 32 Gaussians">

## Use

```bash
git clone https://github.com/arthifact/GMNet.git
cd GMNet
uv sync
```

Download OpenFWI family folders and extract them into `data_raw/`
([links](data_raw/README.md)). Then:

```bash
uv run src/process.py     # once: data_raw/ -> data_processed/
uv run src/train.py       # -> outputs/<name>/model/
uv run src/evaluate.py    # test RMSE and FWI-ready -> outputs/<name>/evaluation/
uv run src/predict.py     # macromodels for your shots -> outputs/<name>/predictions/
```

Every setting is in [config.toml](config.toml). The default band is 0-10 Hz, to compare with other
networks. For the realism check, the 4-10 Hz that field surveys record, set `name = "4-10Hz"` and
`band = [4, 10]`. Each run keeps its own folder ([outputs/README.md](outputs/README.md)).

Needs [uv](https://docs.astral.sh/uv/); training picks CUDA, then MPS, then CPU. Tests:
`uv run tests/test_steps.py`.

## License

[MIT](LICENSE)
