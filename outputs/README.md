# Outputs

One folder per experiment, named by `name` in its config:

```
outputs/0-10Hz/
  config.toml      the config that made this run
  model/           train:    best.pt, history.json
  evaluation/      evaluate: metrics.json, per_case.csv, gaussians.npz
  predictions/     predict:  velocity.npy, gaussians.npz
```

- `best.pt`: the weights at the epoch with the lowest validation error, with the band they were trained on.
- `history.json`: loss, validation and training error and time, per epoch.
- `metrics.json`: mean RMSE, geometry bound and FWI-ready, overall and per family.
- `per_case.csv`: the same scores for every test case.
- `velocity.npy`: the predicted macromodels, `[N, 70, 70]` in m/s.
- `gaussians.npz`: the same models as 32 Gaussians `[a, x, z, log σx, log σz, θ]` and a background `[b0, b1]` each.

Training never overwrites a model: delete the folder or change `name` to retrain. Evaluation and
predictions are rewritten each time. Nothing here except this file is tracked by git.
