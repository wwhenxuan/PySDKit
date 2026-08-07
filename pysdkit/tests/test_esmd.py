# -*- coding: utf-8 -*-
"""
Unit tests for Extreme-Point Symmetric Mode Decomposition (ESMD).

References
----------
[1] J. L. Wang and Z. J. Li, "Extreme-Point Symmetric Mode Decomposition
    Method for Data Analysis," Adv. Adapt. Data Anal., 2013.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pysdkit import ESMD, esmd
from pysdkit._esmd.esmd import (
    find_extrema,
    midpoints_from_extrema,
    mean_curve,
    make_esmd_example3,
    load_wind_demo,
    instantaneous_amplitude,
    instantaneous_frequency,
)


class ESMDHelpersTest(unittest.TestCase):
    def test_find_extrema_sinusoid(self):
        t = np.linspace(0.0, 2.0 * np.pi, 401)
        x = np.sin(t)
        idx, vals = find_extrema(x)
        self.assertGreaterEqual(len(idx), 1)
        self.assertEqual(idx.shape, vals.shape)
        self.assertTrue(np.all(np.diff(idx) > 0))
        self.assertTrue(np.any(vals > 0.5))
        self.assertTrue(np.any(vals < -0.5))

    def test_midpoints_and_mean_curve(self):
        t = np.arange(200, dtype=float)
        x = np.sin(0.2 * t) + 0.3 * np.sin(0.05 * t)
        idx, vals = find_extrema(x)
        mid_t, mid_y = midpoints_from_extrema(t[idx], vals)
        self.assertGreaterEqual(len(mid_t), 2)
        m = mean_curve(x, t, n_curves=2)
        self.assertEqual(m.shape, x.shape)
        self.assertTrue(np.isfinite(m).all())

    def test_make_esmd_example3_components(self):
        demo = make_esmd_example3(n=400)
        recon = demo["mode1"] + demo["mode2"] + demo["trend"]
        self.assertTrue(np.allclose(demo["signal"], recon, atol=1e-12))
        self.assertEqual(demo["signal"].shape, (400,))


class ESMDAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.demo = make_esmd_example3(n=400)
        self.signal = self.demo["signal"]
        self.dt = float(self.demo["dt"])

    def test_call_and_fit_transform_match(self):
        decomp = ESMD(n_curves=2, max_sift=12, optimize_sift=False, max_imfs=6)
        out1 = decomp.fit_transform(self.signal, dt=self.dt, compute_di=False)
        out2 = decomp(self.signal, dt=self.dt)
        self.assertEqual(out1.shape, out2.shape)
        self.assertTrue(np.allclose(out1, out2))

    def test_functional_api(self):
        imfs = esmd(
            self.signal,
            dt=self.dt,
            n_curves=2,
            max_sift=10,
            optimize_sift=False,
        )
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], self.signal.size)

    def test_reconstruction(self):
        decomp = ESMD(n_curves=2, max_sift=15, optimize_sift=False)
        imfs = decomp.fit_transform(self.signal, dt=self.dt, compute_di=False)
        recon = np.sum(imfs, axis=0)
        self.assertTrue(np.allclose(self.signal, recon, atol=1e-10))

    def test_example3_mode_recovery(self):
        # Paper Example 3: two oscillatory modes + quadratic trend.
        decomp = ESMD(
            n_curves=2,
            max_sift=15,
            optimize_sift=False,
            extreme_num_r=4,
            max_imfs=8,
        )
        imfs = decomp.fit_transform(self.signal, dt=self.dt, compute_di=False)
        self.assertGreaterEqual(imfs.shape[0], 2)

        residual = imfs[-1]
        trend_corr = abs(float(np.corrcoef(self.demo["trend"], residual)[0, 1]))
        self.assertGreater(trend_corr, 0.90)

        mode_refs = (self.demo["mode1"], self.demo["mode2"])
        best_corrs = []
        for ref in mode_refs:
            best = max(
                abs(float(np.corrcoef(ref, imfs[k])[0, 1]))
                for k in range(imfs.shape[0] - 1)
            )
            best_corrs.append(best)
        self.assertGreater(best_corrs[0], 0.90)
        self.assertGreater(best_corrs[1], 0.80)

    def test_optimize_sift_selects_k(self):
        decomp = ESMD(
            n_curves=2,
            min_sift=5,
            max_sift=20,
            optimize_sift=True,
            extreme_num_r=4,
        )
        imfs = decomp.fit_transform(self.signal, dt=self.dt, compute_di=False)
        self.assertIsNotNone(decomp.opt_sift_)
        self.assertIsNotNone(decomp.variance_ratios_)
        assert decomp.opt_sift_ is not None
        assert decomp.variance_ratios_ is not None
        self.assertGreaterEqual(decomp.opt_sift_, 5)
        self.assertLessEqual(decomp.opt_sift_, 20)
        self.assertEqual(decomp.variance_ratios_.size, 16)
        self.assertTrue(np.isfinite(decomp.variance_ratios_).all())
        self.assertEqual(imfs.shape[1], self.signal.size)
        self.assertTrue(np.allclose(self.signal, np.sum(imfs, axis=0), atol=1e-10))

    def test_get_variance_ratio(self):
        decomp = ESMD(
            n_curves=2,
            min_sift=5,
            max_sift=12,
            extreme_num_r=4,
        )
        ks, ratios = decomp.get_variance_ratio(self.signal, dt=self.dt)
        self.assertEqual(ks.tolist(), list(range(5, 13)))
        self.assertEqual(ratios.shape, ks.shape)
        self.assertTrue(np.all(ratios > 0.0))
        self.assertTrue(np.all(ratios <= 1.0 + 1e-8))
        self.assertEqual(decomp.opt_sift_, int(ks[int(np.argmin(ratios))]))

    def test_esmd_i_and_iii(self):
        for n_curves in (1, 3):
            decomp = ESMD(
                n_curves=n_curves,
                max_sift=8,
                optimize_sift=False,
                max_imfs=5,
            )
            imfs = decomp.fit_transform(self.signal, dt=self.dt, compute_di=False)
            self.assertEqual(imfs.shape[1], self.signal.size)
            self.assertTrue(np.allclose(self.signal, np.sum(imfs, axis=0), atol=1e-10))

    def test_direct_interpolation_outputs(self):
        decomp = ESMD(n_curves=2, max_sift=12, optimize_sift=False, max_imfs=6)
        imfs = decomp.fit_transform(self.signal, dt=self.dt, compute_di=True)
        self.assertIsNotNone(decomp.amplitudes_)
        self.assertIsNotNone(decomp.frequencies_)
        self.assertIsNotNone(decomp.energy_)
        assert decomp.amplitudes_ is not None
        assert decomp.frequencies_ is not None
        assert decomp.energy_ is not None
        n_osc = imfs.shape[0] - 1
        self.assertEqual(decomp.amplitudes_.shape, (n_osc, self.signal.size))
        self.assertEqual(decomp.frequencies_.shape, (n_osc, self.signal.size))
        self.assertEqual(decomp.energy_.shape, (self.signal.size,))
        self.assertTrue(np.all(decomp.amplitudes_ >= 0.0))
        self.assertTrue(np.all(decomp.energy_ >= 0.0))

    def test_di_amplitude_frequency_helpers(self):
        t = np.linspace(0.0, 2.0, 400)
        x = np.sin(2.0 * np.pi * 4.0 * t)
        amp = instantaneous_amplitude(x, t)
        freq = instantaneous_frequency(x, t)
        self.assertEqual(amp.shape, x.shape)
        self.assertTrue(np.mean(amp) > 0.7)
        self.assertTrue(np.mean(freq) > 2.0)

    def test_invalid_inputs(self):
        decomp = ESMD(optimize_sift=False, max_sift=5)
        with self.assertRaises(ValueError):
            decomp.fit_transform(np.arange(5.0))
        with self.assertRaises(ValueError):
            decomp.fit_transform(np.arange(20.0), dt=0.0)
        with self.assertRaises(ValueError):
            ESMD(n_curves=0)

    def test_str(self):
        text = str(ESMD(n_curves=2, max_sift=30))
        self.assertIn("ESMD", text)


class ESMDWindDemoTest(unittest.TestCase):
    def test_load_and_decompose_wind_file(self):
        # Synthetic CSV with the same layout as a wind-demo series.
        import tempfile

        rng = np.random.default_rng(0)
        n = 300
        series = (
            2.0 * np.sin(2.0 * np.pi * np.arange(n) / 40.0)
            + 0.8 * np.sin(2.0 * np.pi * np.arange(n) / 12.0)
            + 0.2 * rng.normal(size=n)
        )
        with tempfile.TemporaryDirectory() as tmp:
            wind_path = Path(tmp) / "winddata.txt"
            # three columns; loader uses column 0
            table = np.column_stack([series, series * 0.1, series * 0.01])
            np.savetxt(wind_path, table, delimiter=",")

            demo = load_wind_demo(str(wind_path))
            self.assertEqual(demo["signal"].size, n)
            self.assertAlmostEqual(demo["dt"], 0.05)
            self.assertTrue(np.allclose(demo["signal"], series))

            decomp = ESMD(
                n_curves=2,
                min_sift=5,
                max_sift=12,
                optimize_sift=True,
                extreme_num_r=4,
                max_imfs=8,
            )
            imfs = decomp.fit_transform(demo["signal"], dt=demo["dt"], compute_di=True)
            self.assertGreaterEqual(imfs.shape[0], 2)
            self.assertTrue(
                np.allclose(demo["signal"], np.sum(imfs, axis=0), atol=1e-8)
            )
            self.assertIsNotNone(decomp.opt_sift_)
            self.assertIsNotNone(decomp.variance_ratios_)
            assert decomp.variance_ratios_ is not None
            self.assertTrue(np.isfinite(decomp.variance_ratios_).all())
            self.assertTrue(np.all(decomp.variance_ratios_ > 0.0))


if __name__ == "__main__":
    unittest.main()
