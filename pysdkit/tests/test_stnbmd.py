# -*- coding: utf-8 -*-
"""
Automated tests for Short-Time Narrow-Banded Mode Decomposition (STNBMD).
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import STNBMD, stnbmd
from pysdkit._vncmd.stnbmd import (
    ps90,
    ps90f,
    analytic_signal,
    first_difference_matrix,
    second_difference_matrix,
    build_smoothing_filters,
    schedule_index,
    fft_two_to_one,
    instantaneous_frequency,
    make_order_tracking_demo,
    constant_frequency_init,
    stnbm_decomp_ig,
)


class HilbertHelperTest(unittest.TestCase):
    def test_ps90f_shape_even_odd(self) -> None:
        x = np.random.randn(64, 2)
        y = ps90f(x)
        self.assertEqual(y.shape, x.shape)
        y2 = ps90f(x[:, 0])
        self.assertEqual(y2.shape, (64,))

        x_odd = np.random.randn(63)
        y_odd = ps90f(x_odd)
        self.assertEqual(y_odd.shape, (63,))

    def test_ps90f_custom_n(self) -> None:
        x = np.random.randn(50)
        y = ps90f(x, n=64)
        self.assertEqual(y.shape, (50,))

    def test_ps90_sine_to_neg_cosine(self) -> None:
        n = 512
        t = np.arange(n) / float(n)
        x = np.sin(2 * np.pi * 8 * t)
        y = ps90(x)
        c = np.cos(2 * np.pi * 8 * t)
        corr = np.corrcoef(y, -c)[0, 1]
        self.assertGreater(corr, 0.99)

    def test_analytic_signal(self) -> None:
        x = np.random.randn(100)
        z = analytic_signal(x)
        self.assertTrue(np.iscomplexobj(z))
        np.testing.assert_allclose(np.real(z), x)


class OperatorHelperTest(unittest.TestCase):
    def test_first_difference_matrix(self) -> None:
        nt, fs = 10, 100.0
        d1 = first_difference_matrix(nt, fs)
        self.assertEqual(d1.shape, (nt - 1, nt))
        x = np.arange(nt, dtype=float)
        dx = d1 @ x
        np.testing.assert_allclose(dx, fs * np.diff(x))

    def test_second_difference_matrix(self) -> None:
        nt, fs = 12, 50.0
        d2 = second_difference_matrix(nt, fs)
        self.assertEqual(d2.shape, (nt - 2, nt))
        x = np.arange(nt, dtype=float) ** 2
        # forward second difference: x[i] - 2 x[i+1] + x[i+2]
        expected = (fs**2) * (x[:-2] - 2 * x[1:-1] + x[2:])
        np.testing.assert_allclose(d2 @ x, expected)

    def test_build_smoothing_filters(self) -> None:
        f1, f2 = build_smoothing_filters(32, 100.0, [0.1, 0.01], [1.0, 0.1])
        self.assertEqual(len(f1), 2)
        self.assertEqual(f1[0].shape, (32, 32))
        self.assertEqual(f2[1].shape, (32, 32))
        # F should be symmetric SPD-ish (positive eigenvalues)
        eig = np.linalg.eigvalsh(f1[0])
        self.assertTrue(np.all(eig > 0))

    def test_build_smoothing_filters_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            build_smoothing_filters(16, 10.0, [0.1], [0.1, 0.01])

    def test_schedule_index(self) -> None:
        abitr = np.array([20.0, 50.0, 200.0])
        self.assertEqual(schedule_index(1, abitr), 0)
        self.assertEqual(schedule_index(20, abitr), 0)
        self.assertEqual(schedule_index(21, abitr), 1)
        self.assertEqual(schedule_index(50, abitr), 1)
        self.assertEqual(schedule_index(51, abitr), 2)
        self.assertEqual(schedule_index(500, abitr), 2)


class SpectrumHelperTest(unittest.TestCase):
    def test_fft_two_to_one_even(self) -> None:
        nt = 128
        fs = 100.0
        x = np.sin(2 * np.pi * 5 * np.arange(nt) / fs)
        xf2 = np.fft.fft(x)
        xf1, f1 = fft_two_to_one(xf2, fs, nt)
        self.assertEqual(xf1.size, f1.size)
        self.assertEqual(f1[0], 0.0)
        peak = f1[np.argmax(np.abs(xf1))]
        self.assertAlmostEqual(peak, 5.0, delta=fs / nt)

    def test_fft_two_to_one_odd_and_matrix(self) -> None:
        x = np.random.randn(65, 2)
        xf2 = np.fft.fft(x, axis=0)
        xf1, f1 = fft_two_to_one(xf2, 50.0, 65)
        self.assertEqual(xf1.shape[1], 2)
        self.assertEqual(xf1.shape[0], f1.size)

    def test_instantaneous_frequency(self) -> None:
        fs = 100.0
        t = np.arange(200) / fs
        phz = 2 * np.pi * 7 * t
        ifrq = instantaneous_frequency(phz, fs)
        np.testing.assert_allclose(ifrq, 7.0, atol=1e-10)

        phz2 = np.column_stack([phz, 2 * phz])
        ifrq2 = instantaneous_frequency(phz2, fs)
        self.assertEqual(ifrq2.shape, (199, 2))


class DemoFactoryTest(unittest.TestCase):
    def test_make_order_tracking_demo(self) -> None:
        demo = make_order_tracking_demo(fs=100.0, nt=500)
        self.assertEqual(demo["signal"].shape, (500,))
        self.assertEqual(demo["modes"].shape, (500, 3))
        self.assertEqual(demo["true_if"].shape, (500, 3))

    def test_constant_frequency_init(self) -> None:
        ampg, phzg = constant_frequency_init(100, 50.0, [2.0, 4.0])
        self.assertEqual(ampg.shape, (100, 2))
        self.assertEqual(phzg.shape, (100, 2))
        np.testing.assert_allclose(np.abs(ampg), 1.0)


class STNBMDAlgoTest(unittest.TestCase):
    def test_str_and_call(self) -> None:
        model = STNBMD(fs=100.0, abitr=[5, 10, 15], tol=1e-4)
        self.assertIn("STNBMD", str(model))
        demo = make_order_tracking_demo(nt=300)
        modes = model(demo["signal"], frequencies=[2.0, 4.0, 6.0])
        self.assertEqual(modes.shape[0], 3)
        self.assertEqual(modes.shape[1], 300)

    def test_functional_matches_class(self) -> None:
        demo = make_order_tracking_demo(nt=400)
        params = dict(
            fs=float(demo["fs"]),
            frequencies=[2.0, 4.0, 6.0],
            alpha=[0.1, 0.01, 0.01],
            beta=[1.0, 0.1, 0.001],
            abitr=[10, 20, 40],
            tol=1e-5,
        )
        xnb1, err1, amp1, phz1 = stnbmd(demo["signal"], **params)
        model = STNBMD(
            fs=params["fs"],
            alpha=params["alpha"],
            beta=params["beta"],
            abitr=params["abitr"],
            tol=params["tol"],
        )
        modes, xnb2, err2, amp2, phz2 = model.fit_transform(
            demo["signal"], frequencies=params["frequencies"], return_all=True
        )
        np.testing.assert_allclose(xnb1, xnb2)
        np.testing.assert_allclose(err1, err2)
        np.testing.assert_allclose(modes, np.real(xnb2).T)

    def test_stnbm_decomp_ig_validation(self) -> None:
        x = np.random.randn(50)
        ampg = np.ones((50, 2), dtype=complex)
        phzg = np.zeros((50, 2))
        with self.assertRaises(ValueError):
            stnbm_decomp_ig(x, 100.0, ampg[:40], phzg, [0.1], [1.0], [10], 1e-6)
        with self.assertRaises(ValueError):
            stnbm_decomp_ig(x, 100.0, ampg, phzg, [0.1], [1.0, 0.1], [10], 1e-6)

    def test_single_mode_tone(self) -> None:
        fs = 200.0
        t = np.arange(400) / fs
        x = np.sin(2 * np.pi * 5 * t)
        model = STNBMD(
            fs=fs,
            alpha=[0.1, 0.01],
            beta=[1.0, 0.01],
            abitr=[20, 60],
            tol=1e-6,
        )
        modes, xnb, err, amp, phz = model.fit_transform(
            x, frequencies=[5.0], return_all=True
        )
        self.assertEqual(modes.shape, (1, 400))
        # reconstruction should be close
        recon = np.real(xnb[:, 0])
        self.assertTrue(np.allclose(recon, x, atol=0.15))
        ifrq = model.instantaneous_frequency_hz()
        self.assertLess(np.abs(np.mean(ifrq) - 5.0), 0.25)

    def test_order_tracking_demo_recon_and_if(self) -> None:
        demo = make_order_tracking_demo()
        model = STNBMD(
            fs=float(demo["fs"]),
            alpha=[1e-1, 1e-2, 1e-2],
            beta=[1.0, 1e-1, 1e-3],
            abitr=[20, 50, 120],
            tol=1e-6,
        )
        modes = model.fit_transform(demo["signal"], frequencies=[2.0, 4.0, 6.0])
        recon = np.sum(modes, axis=0)
        rel = np.linalg.norm(recon - demo["signal"]) / np.linalg.norm(demo["signal"])
        self.assertLess(rel, 0.05)

        ifrq = model.instantaneous_frequency_hz()
        # mean IF should be near mean true upsweep harmonics
        true_mean = demo["true_if"].mean(axis=0)
        est_mean = ifrq.mean(axis=0)
        # allow mode order match by sorting
        np.testing.assert_allclose(np.sort(est_mean), np.sort(true_mean), rtol=0.15)

    def test_if_before_fit_raises(self) -> None:
        with self.assertRaises(ValueError):
            STNBMD().instantaneous_frequency_hz()

    def test_stnbmd_requires_init(self) -> None:
        with self.assertRaises(ValueError):
            stnbmd(np.random.randn(100), fs=50.0)


if __name__ == "__main__":
    unittest.main()
