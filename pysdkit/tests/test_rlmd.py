# -*- coding: utf-8 -*-
"""
Unit tests for Robust Local Mean Decomposition (RLMD).
"""

import unittest

import numpy as np

from pysdkit import RLMD
from pysdkit.data import test_emd, test_univariate_signal
from pysdkit._lmd.rlmd import extr, extend, smooth


def matlab_demo_signal(fs: float = 1000.0, n: int = 3000):
    """
    Downsampled version:

    ``x1 = (2+cos(2*pi*0.5*t)).*cos(2*pi*5*t+15*t.^2)``
    ``x2 = cos(2*pi*2*t)``
    """
    t = (np.arange(1, n + 1, dtype=float)) / fs
    x1 = (2.0 + np.cos(2 * np.pi * 0.5 * t)) * np.cos(2 * np.pi * 5 * t + 15 * t**2)
    x2 = np.cos(2 * np.pi * 2 * t)
    return t, x1 + x2, x1, x2


class RLMDTest(unittest.TestCase):
    """Unit tests for Robust Local Mean Decomposition (RLMD)."""

    def test_fit_transform(self) -> None:
        """Decomposition runs and returns a finite 2-D array of correct length."""
        rlmd = RLMD(max_imfs=5, max_iter=20)
        for index in range(1, 4):
            _, signal = test_univariate_signal(case=index)
            imfs = rlmd.fit_transform(signal)
            self.assertEqual(imfs.ndim, 2)
            self.assertEqual(imfs.shape[1], signal.size)
            self.assertTrue(np.all(np.isfinite(imfs)))
            self.assertGreaterEqual(imfs.shape[0], 1)

    def test_default_call(self) -> None:
        """``__call__`` matches ``fit_transform``."""
        _, signal = test_emd()
        rlmd = RLMD(max_imfs=4, max_iter=20)
        a = rlmd(signal)
        b = rlmd.fit_transform(signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_perfect_reconstruction(self) -> None:
        """Sum of PFs + residue recovers the input."""
        _, signal = test_emd()
        imfs = RLMD(max_imfs=6, max_iter=25).fit_transform(signal)
        self.assertTrue(np.allclose(imfs.sum(0), signal, atol=1e-8))

    def test_max_imfs_cap(self) -> None:
        """Number of PFs (excluding residue) does not exceed ``max_imfs``."""
        _, signal = test_emd()
        for k in [2, 3, 5]:
            imfs = RLMD(max_imfs=k, max_iter=20).fit_transform(signal)
            self.assertLessEqual(imfs.shape[0], k + 1)
            self.assertGreaterEqual(imfs.shape[0], 1)

    def test_return_all(self) -> None:
        """``return_all`` yields PF / AM / FM with matching shapes."""
        _, signal = test_emd()
        pfs, ams, fms = RLMD(max_imfs=5, max_iter=20).fit_transform(
            signal, return_all=True
        )
        self.assertEqual(pfs.shape[1], signal.size)
        self.assertEqual(ams.shape[1], signal.size)
        self.assertEqual(fms.shape[1], signal.size)
        self.assertEqual(ams.shape[0], fms.shape[0])
        self.assertEqual(ams.shape[0], pfs.shape[0] - 1)
        # PF ≈ AM * FM for each extracted mode
        for i in range(ams.shape[0]):
            self.assertTrue(np.allclose(pfs[i], ams[i] * fms[i], atol=1e-10))

    def test_matlab_demo_recovers_modes(self) -> None:
        """MATLAB ``lmd_demo.m`` mixture: both chirp-AM and tone recovered."""
        _, x, x1, x2 = matlab_demo_signal(fs=1000.0, n=2000)
        pfs = RLMD(max_imfs=10, max_iter=30).fit_transform(x)
        self.assertTrue(np.allclose(pfs.sum(0), x, atol=1e-8))
        corr1 = max(
            abs(float(np.corrcoef(x1, pfs[k])[0, 1])) for k in range(pfs.shape[0] - 1)
        )
        corr2 = max(
            abs(float(np.corrcoef(x2, pfs[k])[0, 1])) for k in range(pfs.shape[0] - 1)
        )
        self.assertGreater(corr1, 0.95)
        self.assertGreater(corr2, 0.95)

    def test_extr_sine(self) -> None:
        t = np.linspace(0, 4 * np.pi, 400)
        s = np.sin(t)
        indmin, indmax, _ = extr(s)
        self.assertGreater(indmin.size, 0)
        self.assertGreater(indmax.size, 0)
        self.assertTrue(np.all(s[indmin] <= s[np.clip(indmin + 1, 0, len(s) - 1)]))

    def test_extend_no_op(self) -> None:
        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        imin, imax, _ = extr(x)
        eimin, eimax, ex, cut = extend(x, imin, imax, extd_r=0.0)
        self.assertTrue(np.allclose(ex, x))
        self.assertEqual(list(cut), [0, x.size - 1])

    def test_smooth_odd_span(self) -> None:
        x = np.arange(20, dtype=float)
        y = smooth(x, span=5)
        self.assertEqual(y.size, x.size)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_str(self) -> None:
        self.assertIn("RLMD", str(RLMD()))

    def test_invalid_smooth_mode(self) -> None:
        rlmd = RLMD(smooth_mode="not-a-mode")
        with self.assertRaises(ValueError):
            rlmd.fit_transform(np.sin(np.linspace(0, 10, 200)))


if __name__ == "__main__":
    unittest.main()
