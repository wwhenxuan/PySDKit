# -*- coding: utf-8 -*-
"""
Automated tests for Variational Time-Frequency Mode Tracking Decomposition.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import VTFMTD, vtfmtd
from pysdkit._tfa.vtfmtd import (
    stft,
    frequency_axis,
    bin_index_grid,
    expand_omega_init,
    first_difference_gram,
    estimate_if_centroid,
    smooth_if,
    moving_average_if,
    omega_bins_to_hz,
)
from pysdkit.data import load_dual_signal_noise, load_map2, load_single_nsignal


class VTFMTDHelperTest(unittest.TestCase):
    """Unit tests for helper routines."""

    def test_stft_shape_and_dtype(self) -> None:
        n = 256
        x = np.exp(1j * 2 * np.pi * 40 * np.arange(n) / n)
        g = stft(x, hlength=33)
        self.assertEqual(g.shape, (int(round(n / 2)), n))
        self.assertTrue(np.iscomplexobj(g))

    def test_stft_default_window(self) -> None:
        x = np.random.randn(200) + 1j * np.random.randn(200)
        g = stft(x)
        self.assertEqual(g.shape[1], 200)

    def test_stft_rejects_too_short(self) -> None:
        with self.assertRaises(ValueError):
            stft(np.array([1.0]))

    def test_frequency_axis(self) -> None:
        f = frequency_axis(3000, 3000.0)
        self.assertEqual(f.size, 1500)
        self.assertAlmostEqual(f[0], 0.0)
        self.assertAlmostEqual(f[1], 1.0)

    def test_bin_index_grid(self) -> None:
        grid = bin_index_grid(4, 5)
        self.assertEqual(grid.shape, (4, 5))
        np.testing.assert_allclose(grid[:, 0], [1, 2, 3, 4])

    def test_expand_omega_init_forms(self) -> None:
        f_bins, n_time, k = 10, 20, 2
        a = expand_omega_init(np.array([100.0, 200.0]), f_bins, n_time, k)
        self.assertEqual(a.shape, (f_bins, n_time, k))
        self.assertTrue(np.allclose(a[:, :, 0], 100.0))

        b = expand_omega_init(
            np.ones((n_time, k)) * np.array([3.0, 7.0]), f_bins, n_time, k
        )
        self.assertTrue(np.allclose(b[0, :, 1], 7.0))

        full = np.random.randn(f_bins, n_time, k)
        c = expand_omega_init(full, f_bins, n_time, k)
        np.testing.assert_allclose(c, full)

    def test_expand_omega_init_invalid(self) -> None:
        with self.assertRaises(ValueError):
            expand_omega_init(np.ones((3, 3)), 10, 20, 2)

    def test_first_difference_gram(self) -> None:
        gram = first_difference_gram(5)
        self.assertEqual(gram.shape, (5, 5))
        dense = gram.toarray()
        # D'D should be SPD / positive semi-definite with known corner structure
        self.assertGreater(dense[0, 0], 0.0)
        eig = np.linalg.eigvalsh(dense)
        self.assertTrue(np.all(eig >= -1e-10))

    def test_estimate_if_centroid(self) -> None:
        f_bins, n_time = 8, 16
        grid = bin_index_grid(f_bins, n_time)
        # put all energy at bin index 5 (0-based row 4)
        mode = np.zeros((f_bins, n_time), dtype=complex)
        mode[4, :] = 1.0 + 0.0j
        est = estimate_if_centroid(mode, grid)
        np.testing.assert_allclose(est[0, :], 5.0, atol=1e-12)
        np.testing.assert_allclose(est[-1, :], 5.0, atol=1e-12)

    def test_estimate_if_centroid_zero_energy(self) -> None:
        grid = bin_index_grid(4, 6)
        mode = np.zeros((4, 6), dtype=complex)
        est = estimate_if_centroid(mode, grid)
        np.testing.assert_allclose(est, 0.0)

    def test_smooth_if(self) -> None:
        n_time = 64
        f_bins = 5
        gram = first_difference_gram(n_time)
        traj = 10.0 + 2.0 * np.sin(2 * np.pi * np.arange(n_time) / n_time)
        omega_est = np.tile(traj[None, :], (f_bins, 1))
        beta = 1.0
        sm = smooth_if(omega_est, gram, beta=beta)
        self.assertEqual(sm.shape, (f_bins, n_time))
        np.testing.assert_allclose(sm[0], sm[-1])
        # residual of the normal equation (2/beta * D'D + I) x = y
        eye = np.eye(n_time)
        residual = ((2.0 / beta) * gram.toarray() + eye) @ sm[0] - traj
        self.assertLess(np.linalg.norm(residual), 1e-8)

    def test_moving_average_if(self) -> None:
        x = np.arange(20, dtype=float)
        y = moving_average_if(x, win=5)
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(y[0], np.mean(x[:3]))

    def test_omega_bins_to_hz(self) -> None:
        hz = omega_bins_to_hz(np.array([1.0, 2.0, 3.0]), fs=1000.0, n=1000)
        np.testing.assert_allclose(hz, [0.0, 1.0, 2.0])


class VTFMTDDataTest(unittest.TestCase):
    """Packaged demo loaders."""

    def test_load_dual_signal_noise(self) -> None:
        demo = load_dual_signal_noise()
        self.assertEqual(demo["signal"].shape, (3000,))
        self.assertTrue(np.iscomplexobj(demo["signal"]))
        self.assertEqual(demo["fs"], 3000.0)
        self.assertEqual(demo["K"], 2)

    def test_load_single_nsignal(self) -> None:
        demo = load_single_nsignal()
        self.assertEqual(demo["signal"].shape, (8011,))
        self.assertTrue(np.iscomplexobj(demo["signal"]))
        self.assertEqual(demo["fs"], 8011.0)

    def test_load_map2(self) -> None:
        cmap = load_map2()
        self.assertEqual(cmap.shape, (64, 3))
        self.assertTrue(np.all(cmap >= 0.0) and np.all(cmap <= 1.0))


class VTFMTDAlgoTest(unittest.TestCase):
    """Core algorithm / OOP interface."""

    def test_str_and_call(self) -> None:
        model = VTFMTD(hlength=21, K=1, alpha=1e-5, max_iter=5)
        self.assertIn("VTFMTD", str(model))

        n = 256
        t = np.arange(n) / n
        sig = np.exp(1j * 2 * np.pi * 40 * t)
        gk, omega = model(sig, omega_init=np.array([40.0]))
        self.assertEqual(gk.shape[-1], 1)
        self.assertEqual(omega.shape[-1], 1)

    def test_functional_matches_class(self) -> None:
        n = 512
        fs = float(n)
        t = np.arange(1, n + 1) / fs
        sig = np.exp(
            1j * 2 * np.pi * (80 * t + 0.5 / np.pi * np.sin(2 * np.pi * 5 * t))
        )
        init = np.array([80.0])
        params = dict(
            hlength=25,
            K=1,
            alpha=1e-5,
            sigma=0.01,
            beta=0.5,
            max_iter=30,
            epsilon=1e-3,
        )
        gk1, om1 = vtfmtd(sig, omega_init=init, **params)
        model = VTFMTD(**params)
        gk2, om2 = model.fit_transform(sig, omega_init=init)
        np.testing.assert_allclose(gk1, gk2, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(om1, om2, rtol=1e-10, atol=1e-10)

    def test_return_all_and_if_trajectories(self) -> None:
        n = 256
        sig = np.exp(1j * 2 * np.pi * 30 * np.arange(n) / n)
        model = VTFMTD(hlength=21, K=1, max_iter=10)
        gk, omega, g = model.fit_transform(
            sig, omega_init=np.array([30.0]), return_all=True
        )
        self.assertEqual(g.shape, gk.shape[:2])
        traj = model.if_trajectories()
        self.assertEqual(traj.shape, (1, n))
        np.testing.assert_allclose(traj[0], omega[0, :, 0])

    def test_if_trajectories_before_fit_raises(self) -> None:
        model = VTFMTD()
        with self.assertRaises(ValueError):
            model.if_trajectories()

    def test_dual_component_noiseless_tracks_if(self) -> None:
        """Reproduce the spirit of MATLAB test1.m (noiseless dual tones)."""
        fs = 3000.0
        t = np.arange(1, int(fs) + 1) / fs
        sig = np.exp(
            1j * 2 * np.pi * (250 * t + 0.5 / np.pi * np.sin(2 * np.pi * 20 * t))
        )
        sig = sig + np.exp(
            1j * 2 * np.pi * (950 * t + 0.5 / np.pi * np.sin(2 * np.pi * 60 * t))
        )
        true1 = 250 + 20 * np.cos(40 * np.pi * t)
        true2 = 950 + 60 * np.cos(120 * np.pi * t)

        model = VTFMTD(
            hlength=30,
            K=2,
            alpha=1e-5,
            sigma=0.01,
            beta=1.0,
            max_iter=40,
            epsilon=1e-3,
        )
        _gk, omega = model.fit_transform(sig, omega_init=np.array([250.0, 950.0]))
        est1 = omega[0, :, 0]
        est2 = omega[0, :, 1]
        # allow mode swap
        err_a = min(np.mean(np.abs(est1 - true1)), np.mean(np.abs(est1 - true2)))
        err_b = min(np.mean(np.abs(est2 - true1)), np.mean(np.abs(est2 - true2)))
        self.assertLess(err_a, 15.0)
        self.assertLess(err_b, 40.0)

    def test_packaged_dual_noise_runs(self) -> None:
        demo = load_dual_signal_noise()
        model = VTFMTD(
            hlength=28,
            K=2,
            alpha=5e-5,
            sigma=0.01,
            beta=0.3,
            max_iter=8,
            epsilon=1e-3,
        )
        gk, omega = model.fit_transform(
            demo["signal"], omega_init=np.array([250.0, 950.0])
        )
        self.assertEqual(gk.shape[2], 2)
        self.assertEqual(omega.shape[2], 2)
        # reconstruction of STFT should be finite
        self.assertTrue(np.isfinite(np.abs(gk)).all())

    def test_packaged_single_noise_runs(self) -> None:
        demo = load_single_nsignal()
        # use a shorter run for unit-test speed: downsample-free but fewer iters
        model = VTFMTD(
            hlength=25,
            K=1,
            alpha=1e-8,
            sigma=0.01,
            beta=0.1,
            max_iter=5,
            epsilon=1e-3,
        )
        gk, omega = model.fit_transform(demo["signal"], omega_init=np.array([2000.0]))
        self.assertEqual(gk.shape[0], int(round(demo["signal"].size / 2.0)))
        self.assertEqual(omega.shape[-1], 1)


if __name__ == "__main__":
    unittest.main()
