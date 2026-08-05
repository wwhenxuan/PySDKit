# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 15:54:22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

import numpy as np

from pysdkit import LMD
from pysdkit.data import test_emd, test_univariate_signal


def _pylmd_demo_signal(n: int = 101) -> np.ndarray:
    """Multi-tone signal from the PyLMD package README / ``__main__``."""
    x = np.linspace(0.0, 100.0, n)
    return (
        (2.0 / 3.0) * np.sin(x * 30.0)
        + (2.0 / 3.0) * np.sin(x * 17.5)
        + (4.0 / 5.0) * np.cos(x * 2.0)
    )


class LMDTest(unittest.TestCase):
    """Unit tests for classical Local Mean Decomposition (LMD)."""

    def test_fit_transform(self) -> None:
        """Signal decomposition runs and returns a 2-D array of correct length."""
        lmd = LMD()
        for index in range(1, 4):
            _, signal = test_univariate_signal(case=index)
            imfs = lmd.fit_transform(signal)
            self.assertEqual(imfs.ndim, 2)
            self.assertEqual(imfs.shape[1], signal.size)
            self.assertTrue(np.all(np.isfinite(imfs)))

    def test_default_call(self) -> None:
        """``__call__`` matches ``fit_transform``."""
        _, signal = test_emd()
        lmd = LMD(K=4)
        a = lmd(signal)
        b = lmd.fit_transform(signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_perfect_reconstruction(self) -> None:
        """Sum of PFs + residue recovers the input (up to float error)."""
        _, signal = test_emd()
        imfs = LMD(K=5).fit_transform(signal)
        recon = imfs.sum(axis=0)
        self.assertTrue(np.allclose(recon, signal, atol=1e-10))

    def test_imfs_number(self) -> None:
        """At most ``K`` PFs plus one residue row."""
        _, signal = test_emd()
        for k in [2, 3, 5]:
            imfs = LMD(K=k).fit_transform(signal)
            # residue always appended → between 1 and k+1 rows
            self.assertGreaterEqual(imfs.shape[0], 1)
            self.assertLessEqual(imfs.shape[0], k + 1)
            self.assertEqual(imfs.shape[0], k + 1)  # test_emd needs full K PFs

    def test_k_override_does_not_mutate(self) -> None:
        """Passing ``K=`` to ``fit_transform`` must not change ``self.K``."""
        lmd = LMD(K=5)
        _, signal = test_emd()
        _ = lmd.fit_transform(signal, K=2)
        self.assertEqual(lmd.K, 5)

    def test_find_extrema_constant(self) -> None:
        """Constant signals yield endpoint extrema when ``endpoints=True``."""
        extrema = LMD(endpoints=True).find_extrema(np.ones(32))
        self.assertEqual(list(extrema), [0, 31])

    def test_find_extrema_sine(self) -> None:
        t = np.linspace(0, 4 * np.pi, 200)
        s = np.sin(t)
        extrema = LMD(endpoints=True).find_extrema(s)
        self.assertGreaterEqual(extrema.size, 4)
        self.assertEqual(extrema[0], 0)
        self.assertEqual(extrema[-1], s.size - 1)

    def test_monotone_returns_residue_only(self) -> None:
        mono = np.linspace(-1.0, 1.0, 64)
        imfs = LMD().fit_transform(mono)
        self.assertEqual(imfs.shape, (1, 64))
        self.assertTrue(np.allclose(imfs[0], mono))

    def test_local_mean_envelope_shapes(self) -> None:
        t = np.linspace(0, 2 * np.pi, 128)
        s = np.sin(3 * t)
        lmd = LMD()
        extrema = lmd.find_extrema(s)
        m0, m, a0, a = lmd.local_mean_and_envelope(s, extrema)
        self.assertEqual(m0.shape, s.shape)
        self.assertEqual(m.shape, s.shape)
        self.assertEqual(a0.shape, s.shape)
        self.assertEqual(a.shape, s.shape)
        self.assertTrue(np.all(a > 0))

    def test_pylmd_demo_decomposes(self) -> None:
        """PyLMD multi-tone example yields multiple PFs and exact reconstruction."""
        signal = _pylmd_demo_signal(101)
        imfs = LMD(K=8).fit_transform(signal)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.allclose(imfs.sum(0), signal, atol=1e-10))

    def test_str(self) -> None:
        self.assertIn("LMD", str(LMD()))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            LMD().fit_transform(np.ones(3))


if __name__ == "__main__":
    unittest.main()
