# -*- coding: utf-8 -*-
"""
Created on 2026/07/31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Unit tests for Variational Nonlinear Chirp Mode Decomposition (VNCMD).
"""

import unittest
import numpy as np

from pysdkit import VNCMD


def _noise_free_crossing_chirps(fs: float = 1000.0, duration: float = 1.0):
    """
    Scaled version of the MATLAB 'Noisy_free example' (intersecting IFs).

    Returns time, mixture, true modes, true IFs.
    """
    t = np.arange(0, duration + 0.5 / fs, 1.0 / fs)
    # Keep the same IF law as the MATLAB demo (valid for duration=1)
    sig1 = (1 + 0.5 * np.cos(2 * np.pi * t)) * np.cos(
        2 * np.pi * (0.2 + 532 * t - 474 * t**2 + 369 * t**3)
    )
    if1 = 532 - 948 * t + 1107 * t**2
    sig2 = (1 + 0.5 * np.cos(2 * np.pi * t)) * np.cos(
        2 * np.pi * (0.8 + 50 * t + 525 * t**2 - 300 * t**3)
    )
    if2 = 50 + 1050 * t - 900 * t**2
    return t, sig1 + sig2, sig1, sig2, if1, if2


class VNCMDTest(unittest.TestCase):
    """Automated tests for VNCMD"""

    def test_fit_transform_shapes(self) -> None:
        """Output shapes of modes / IF / IA must be (K, N)."""
        fs = 500.0
        t = np.arange(0, 0.5, 1.0 / fs)
        signal = np.cos(2 * np.pi * 40 * t) + np.cos(2 * np.pi * 80 * t)
        eIF = np.vstack([45 * np.ones(t.size), 85 * np.ones(t.size)])

        vncmd = VNCMD(fs=fs, alpha=1e-4, beta=1e-7, var=0.0, tol=1e-5, max_iter=40)
        modes, ifs, ia = vncmd.fit_transform(signal, eIF=eIF)

        self.assertEqual(modes.shape, (2, t.size))
        self.assertEqual(ifs.shape, (2, t.size))
        self.assertEqual(ia.shape, (2, t.size))
        self.assertTrue(np.all(np.isfinite(modes)))
        self.assertTrue(np.all(np.isfinite(ifs)))
        self.assertTrue(np.all(ia >= 0))

    def test_call_interface(self) -> None:
        """Instances must be callable like functions."""
        fs = 400.0
        t = np.arange(0, 0.4, 1.0 / fs)
        signal = np.cos(2 * np.pi * 30 * t) + 0.8 * np.cos(2 * np.pi * 70 * t)
        eIF = np.vstack([35 * np.ones(t.size), 75 * np.ones(t.size)])
        vncmd = VNCMD(fs=fs, alpha=1e-4, beta=1e-7, var=0.0, tol=1e-5, max_iter=30)
        modes, ifs, ia = vncmd(signal, eIF=eIF)
        self.assertEqual(modes.shape[0], 2)
        self.assertEqual(ifs.shape[1], t.size)
        self.assertEqual(ia.shape, modes.shape)

    def test_return_all_histories(self) -> None:
        """return_all=True should provide IF/mode iteration histories."""
        fs = 400.0
        t = np.arange(0, 0.3, 1.0 / fs)
        signal = np.cos(2 * np.pi * 25 * t) + np.cos(2 * np.pi * 60 * t)
        eIF = np.vstack([30 * np.ones(t.size), 65 * np.ones(t.size)])
        vncmd = VNCMD(fs=fs, alpha=1e-4, beta=1e-7, var=0.0, tol=1e-5, max_iter=25)
        if_hist, mode_hist, ia = vncmd.fit_transform(signal, eIF=eIF, return_all=True)
        self.assertEqual(len(if_hist.shape), 3)
        self.assertEqual(len(mode_hist.shape), 3)
        self.assertEqual(if_hist.shape[:2], (2, t.size))
        self.assertEqual(mode_hist.shape[:2], (2, t.size))
        self.assertGreaterEqual(if_hist.shape[-1], 2)
        self.assertEqual(ia.shape, (2, t.size))

    def test_requires_fs_and_eIF(self) -> None:
        """Missing fs or eIF must raise ValueError."""
        signal = np.random.randn(100)
        vncmd = VNCMD(alpha=1e-4, beta=1e-7, var=0.0)
        with self.assertRaises(ValueError):
            vncmd.fit_transform(signal)

        vncmd_fs = VNCMD(fs=100.0, alpha=1e-4, beta=1e-7, var=0.0)
        with self.assertRaises(ValueError):
            vncmd_fs.fit_transform(signal)

    def test_signal_length_must_match_eIF(self) -> None:
        """Signal length and eIF length must be consistent."""
        vncmd = VNCMD(fs=100.0, alpha=1e-4, beta=1e-7, var=0.0, max_iter=5)
        signal = np.random.randn(50)
        eIF = np.ones((2, 40))
        with self.assertRaises(ValueError):
            vncmd.fit_transform(signal, eIF=eIF)

    def test_projec_and_differ_helpers(self) -> None:
        """Low-level helpers must match MATLAB projec / Differ behaviour."""
        # projec: zero variance forces the zero vector when norm > 0
        u = VNCMD.projec(np.array([3.0, 4.0]), var=0.0)
        self.assertTrue(np.allclose(u, 0.0))

        # projec: inside the ball -> unchanged
        v = np.array([0.1, -0.2, 0.05])
        self.assertTrue(np.allclose(VNCMD.projec(v, var=1.0), v))

        vncmd = VNCMD(fs=10.0)
        y = np.array([0.0, 1.0, 4.0, 9.0], dtype=float)
        dy = vncmd.differ(y, delta=1.0)
        # endpoints: forward/backward; interior: central differences
        self.assertAlmostEqual(dy[0], 1.0)
        self.assertAlmostEqual(dy[1], (4.0 - 0.0) / 2.0)
        self.assertAlmostEqual(dy[2], (9.0 - 1.0) / 2.0)
        self.assertAlmostEqual(dy[3], 5.0)

    def test_noise_free_if_accuracy(self) -> None:
        """
        Reproduce the MATLAB noise-free demo (fs=2000 Hz) and check IF accuracy.

        Relative IF errors should be very small (typically < 1%).
        """
        # Nyquist must exceed max IF (~700 Hz) -> use the original MATLAB rate
        fs = 2000.0
        t, sig, sig1, sig2, if1, if2 = _noise_free_crossing_chirps(fs=fs, duration=1.0)
        eIF = np.vstack([700 * np.ones(t.size), 20 * np.ones(t.size)])

        vncmd = VNCMD(fs=fs, alpha=5e-6, beta=1e-6, var=0.0, tol=1e-8, max_iter=300)
        modes, ifs, ia = vncmd.fit_transform(sig, eIF=eIF)

        re1 = np.linalg.norm(ifs[0] - if1) / np.linalg.norm(if1)
        re2 = np.linalg.norm(ifs[1] - if2) / np.linalg.norm(if2)
        self.assertLess(re1, 0.05, msg=f"IF1 relative error too large: {re1}")
        self.assertLess(re2, 0.05, msg=f"IF2 relative error too large: {re2}")

        # Mode reconstruction should also be reasonably accurate
        e1 = np.linalg.norm(modes[0] - sig1) / np.linalg.norm(sig1)
        e2 = np.linalg.norm(modes[1] - sig2) / np.linalg.norm(sig2)
        self.assertLess(e1, 0.15, msg=f"Mode1 relative error too large: {e1}")
        self.assertLess(e2, 0.15, msg=f"Mode2 relative error too large: {e2}")
        self.assertTrue(np.all(ia > 0))

    def test_does_not_mutate_input_eIF(self) -> None:
        """fit_transform must not modify the caller's initial IF array."""
        fs = 300.0
        t = np.arange(0, 0.4, 1.0 / fs)
        signal = np.cos(2 * np.pi * 20 * t) + np.cos(2 * np.pi * 55 * t)
        eIF = np.vstack([25 * np.ones(t.size), 60 * np.ones(t.size)])
        eIF_copy = eIF.copy()
        vncmd = VNCMD(fs=fs, alpha=1e-4, beta=1e-7, var=0.0, tol=1e-5, max_iter=20)
        vncmd.fit_transform(signal, eIF=eIF)
        self.assertTrue(np.allclose(eIF, eIF_copy))

    def test_str(self) -> None:
        """String representation should mention VNCMD."""
        self.assertIn("VNCMD", str(VNCMD(fs=1.0)))


if __name__ == "__main__":
    unittest.main()
