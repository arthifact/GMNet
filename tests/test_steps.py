"""Checks of the representation, the bands, the input scale, the mirror, travel times, the network,
the config and the four steps.

    uv run tests/test_steps.py
"""
import inspect
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from augment import mirror_shots  # noqa: E402
from data import normalize  # noqa: E402
from evaluate import evaluate  # noqa: E402
from gaussians import gaussians, render, solve  # noqa: E402
from model import GMNet  # noqa: E402
from predict import predict  # noqa: E402
import config  # noqa: E402
from process import downsample, field_band, process  # noqa: E402
import train  # noqa: E402
from traveltime import first_arrivals  # noqa: E402


def random_gaussians(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return gaussians(600 * torch.randn(batch, 32, generator=g, dtype=torch.float64),
                     torch.randn(batch, 32, 5, generator=g, dtype=torch.float64))


def fake_openfwi(root, families=('flatvel-a', 'style-a'), files=3, per_file=4):
    """OpenFWI-shaped random data; three files per family, split into one train, validation and test file.
    Style_A is laid out as downloaded from OpenFWI's Drive (Style_A/data, Style_A/model), flatvel-a as one folder."""
    rng = np.random.default_rng(0)
    for family in families:
        drive = root / 'Style_A'
        shots, models = (drive / 'data', drive / 'model') if family == 'style-a' else (root / family,) * 2
        shots.mkdir(parents=True)
        models.mkdir(parents=True, exist_ok=True)
        for k in range(1, files + 1):
            np.save(shots / f'data{k}.npy', rng.normal(0, .5, (per_file, 5, 1000, 70)).astype(np.float32))
            depth = np.linspace(1800, 4200, 70)[None, None, :, None] + rng.normal(0, 200, (per_file, 1, 1, 1))
            np.save(models / f'model{k}.npy', np.broadcast_to(depth, (per_file, 1, 70, 70)).astype(np.float32))


class RepresentationTests(unittest.TestCase):
    def test_any_raw_output_is_a_valid_gaussian(self):
        g = gaussians(torch.zeros(1000, 32), 10 * torch.randn(1000, 32, 5))
        self.assertTrue(((g[..., 1:3] >= 0) & (g[..., 1:3] <= 690)).all())
        self.assertTrue(((g[..., 3:5] >= math.log(50) - 1e-5) & (g[..., 3:5] <= math.log(300) + 1e-5)).all())
        self.assertTrue(((g[..., 3] - g[..., 4]).abs() <= math.log(3) + 1e-5).all())

    def test_amplitudes_and_background_are_linear_and_solved_exactly(self):
        g, trend = random_gaussians(), torch.tensor([[.2, -.3], [-.1, .4]], dtype=torch.float64)
        field = render(g, trend)
        solved = render(*solve(g, field))
        self.assertLess(float((solved - field).square().mean((1, 2)).sqrt().max()), 5.)   # ridge bias only


class DataTests(unittest.TestCase):
    def test_field_band_keeps_4_to_7_hz_only(self):
        t = np.arange(1000) * .001
        for hz, kept in ((2., 0.), (5., 1.), (6.5, 1.), (12., 0.), (30., 0.)):
            tone = np.broadcast_to(np.sin(2 * np.pi * hz * t)[None, None, :, None], (1, 5, 1000, 70))
            band = field_band(tone)
            self.assertEqual(band.shape, (1, 5, 63, 35))
            self.assertAlmostEqual(float(np.abs(band[0, 0, 10:50]).max()), kept, delta=.1)   # leakage of a 1 s tone

    def test_band_can_start_at_0_hz(self):
        t = np.arange(1000) * .001
        for hz, kept in ((1., 1.), (5., 1.), (12., 0.)):
            tone = np.broadcast_to(np.sin(2 * np.pi * hz * t)[None, None, :, None], (1, 5, 1000, 70))
            self.assertAlmostEqual(float(np.abs(field_band(tone, 0, 10)[0, 0, 10:50]).max()), kept, delta=.1)
        with self.assertRaises(ValueError):
            field_band(tone, 4, 40)                                  # above what 16 ms samples hold

    def test_processing_keeps_0_to_31_hz(self):
        t = np.arange(1000) * .001
        for hz, kept in ((1., 1.), (20., 1.), (40., 0.), (60., 0.)):   # 60 Hz would alias without the filter
            tone = np.broadcast_to(np.sin(2 * np.pi * hz * t)[None, None, :, None], (1, 5, 1000, 70))
            self.assertAlmostEqual(float(np.abs(downsample(tone)[0, 0, 10:50]).max()), kept, delta=.15)

    def test_input_ignores_the_amplitude_scale(self):
        shots = torch.randn(2, 5, 63, 35)
        torch.testing.assert_close(normalize(shots), normalize(37. * shots))
        self.assertAlmostEqual(float(torch.sinh(normalize(shots)).square().mean()), 1., places=5)

    def test_mirror_flips_sources_and_receivers(self):
        shots = torch.randn(1, 5, 63, 35)
        for source in range(5):
            torch.testing.assert_close(mirror_shots(shots)[:, source], shots[:, 4 - source].flip(-1), rtol=0, atol=0)


class TravelTimeTests(unittest.TestCase):
    def test_first_arrivals_match_exact_answers(self):
        self.assertAlmostEqual(float(first_arrivals(np.full((70, 70), 2000.))[0, 69]), .345, places=9)
        v0, g, x = 1500., 2., np.arange(1, 70) * 10.          # v = v0 + g z: circular rays
        exact = np.arccosh(1 + g ** 2 * x ** 2 / (2 * v0 ** 2)) / g
        times = first_arrivals(v0 + g * np.arange(70.)[:, None] * 10. * np.ones((1, 70)))[0, 1:]
        self.assertLess(float(np.max(np.abs(times - exact) / exact)), .01)


class ModelTests(unittest.TestCase):
    def test_194_numbers_starting_from_the_lattice(self):
        net = GMNet().eval()
        with torch.no_grad():
            g, trend = net(torch.randn(2, 5, 63, 35))
        self.assertEqual(g[0].numel() + trend[0].numel(), 194)
        torch.testing.assert_close(g, gaussians(torch.zeros(1, 32), net.start[None]).expand(2, -1, -1))
        torch.testing.assert_close(render(g, trend), torch.full((2, 70, 70), 3000.))   # zero heads: 3000 m/s

    def test_any_lattice(self):
        for nx, nz in ((2, 4), (3, 6), (5, 10)):
            with torch.no_grad():
                g, _ = GMNet((nx, nz)).eval()(torch.randn(1, 5, 63, 35))
            self.assertEqual(g.shape[1], nx * nz)
            self.assertTrue(torch.isfinite(g).all())

    def test_gradients_are_finite(self):
        net = GMNet().train()
        train.field_loss(*net(torch.randn(2, 5, 63, 35)), torch.full((2, 70, 70), 2800.)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))


class StepsTest(unittest.TestCase):
    def test_config_fits_every_step(self):
        c = config.load(Path(__file__).resolve().parents[1] / 'config.toml')
        self.assertEqual(c['run'], Path('outputs') / c['name'])
        for step, section in ((train.train, 'train'), (predict, 'predict')):
            inspect.signature(step).bind_partial(**c[section])          # every setting is a parameter
        self.assertEqual(set(c['data']), {'raw', 'processed', 'test', 'validation'})

    def test_process_train_evaluate_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fake_openfwi(tmp / 'data_raw')
            counts = process(tmp / 'data_raw', tmp / 'data_processed')['counts']
            self.assertEqual(counts, dict(train=8, validation=8, test=8))
            history = train.train(tmp / 'outputs/test', tmp / 'data_processed', band=[4, 10], epochs=2, batch_size=4,
                                  device='cpu', config_text='name = "test"\n')
            self.assertEqual(len(history), 2)
            self.assertEqual(torch.load(tmp / 'outputs/test/model/best.pt', weights_only=True)['band_hz'], [4., 10.])
            self.assertTrue((tmp / 'outputs/test/config.toml').exists())
            with self.assertRaises(FileExistsError):                                  # never overwrites a run
                train.train(tmp / 'outputs/test', tmp / 'data_processed', epochs=1, device='cpu')
            summary = evaluate(tmp / 'outputs/test', tmp / 'data_processed', 'test', device='cpu')
            self.assertEqual(summary['count'], 8)
            self.assertTrue(np.isfinite(summary['overall']['rmse_m_s']))
            self.assertTrue(0 <= summary['overall']['fwi_ready'] <= 1)
            g, trend = predict(tmp / 'outputs/test', tmp / 'data_raw/Style_A/data/data3.npy', device='cpu')
            self.assertEqual((tuple(g.shape), tuple(trend.shape)), ((4, 32, 6), (4, 2)))
            self.assertEqual(np.load(tmp / 'outputs/test/predictions/velocity.npy').shape, (4, 70, 70))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
