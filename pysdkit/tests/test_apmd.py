# -*- coding: utf-8 -*-
"""
Unit tests for Adaptive Polymorphic Mode Decomposition (APMD).
"""
import unittest
import numpy as np

from pysdkit import APMD
from pysdkit._apmd.apmd import gausswin, matlab_smooth, nextpow2, optwin, sub_tfrstft


def _make_two_chirp(fs: float = 400.0, T: float = 0.4, seed: int = 0):
    """Simple two-component AM-FM mixture for fast tests."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, 1.0 / fs)
    x1 = (1.0 + 0.2 * np.cos(2 * np.pi * 2.0 * t)) * np.cos(2 * np.pi * 45.0 * t)
    x2 = np.cos(2 * np.pi * (70.0 * t + 25.0 * t**2))
    signal = x1 + x2 + 0.02 * rng.standard_normal(t.size)
    return t, signal, x1, x2


class APMDTest(unittest.TestCase):
    """Automated tests for APMD."""

    def test_fit_transform_shapes(self) -> None:
        """Modes and metadata should have consistent shapes."""
        t, signal, _, _ = _make_two_chirp()
        apmd = APMD(n_modes=4, d=8, thd=0.1, thd2=0.1, max_inner_iter=2)
        modes = apmd.fit_transform(signal, fs=400.0)

        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], signal.size)
        self.assertGreaterEqual(modes.shape[0], 1)
        self.assertLessEqual(modes.shape[0], 4)
        self.assertEqual(apmd.wl_opt.size, modes.shape[0])
        self.assertEqual(apmd.voh.size, modes.shape[0])
        self.assertEqual(apmd.if_mode.shape, modes.shape)
        self.assertEqual(apmd.it_mode.shape, modes.shape)
        self.assertEqual(apmd.tfr_squeezed.shape[1], signal.size)

    def test_return_all_dict(self) -> None:
        """return_all should expose the full result dictionary."""
        t, signal, _, _ = _make_two_chirp()
        apmd = APMD(n_modes=3, d=8, thd=0.15, thd2=0.1, max_inner_iter=2)
        out = apmd.fit_transform(signal, fs=400.0, return_all=True)

        self.assertIsInstance(out, dict)
        for key in (
            "modes",
            "tfr_modes",
            "tfr_squeezed",
            "if_mode",
            "it_mode",
            "u_mode",
            "v_mode",
            "wl_opt",
            "voh",
            "elapsed",
        ):
            self.assertIn(key, out)
        self.assertEqual(out["modes"].shape, apmd.modes.shape)

    def test_call_matches_fit_transform(self) -> None:
        """__call__ should match fit_transform."""
        t, signal, _, _ = _make_two_chirp()
        apmd = APMD(n_modes=3, d=8, thd=0.15, thd2=0.1, max_inner_iter=2, fs=400.0)
        a = apmd(signal)
        b = apmd.fit_transform(signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_missing_fs_raises(self) -> None:
        """Sampling frequency is required."""
        signal = np.random.randn(64)
        apmd = APMD(n_modes=2, max_inner_iter=1)
        with self.assertRaises(ValueError):
            apmd.fit_transform(signal)

    def test_reconstruction_includes_residual(self) -> None:
        """Sum of returned modes (including residual) reconstructs the signal."""
        t, signal, _, _ = _make_two_chirp()
        apmd = APMD(n_modes=4, d=8, thd=0.1, thd2=0.1, max_inner_iter=2)
        modes = apmd.fit_transform(signal, fs=400.0)
        recon = modes.sum(axis=0)
        recon_re = np.linalg.norm(recon - signal) / np.linalg.norm(signal)
        self.assertLess(recon_re, 1e-10)

    def test_mode_energy_reasonable(self) -> None:
        """At least one extracted mode should capture a sizable fraction of energy."""
        t, signal, x1, x2 = _make_two_chirp(fs=400.0, T=0.5)
        apmd = APMD(n_modes=5, d=10, thd=0.05, thd2=0.05, max_inner_iter=4)
        modes = apmd.fit_transform(signal, fs=400.0)

        self.assertTrue(np.all(np.isfinite(modes)))
        cands = modes[:-1] if modes.shape[0] > 1 else modes
        best1 = min(np.linalg.norm(c - x1) / np.linalg.norm(x1) for c in cands)
        best2 = min(np.linalg.norm(c - x2) / np.linalg.norm(x2) for c in cands)
        self.assertLess(best1, 0.75, msg=f"mode1 RE too large: {best1}")
        self.assertLess(best2, 0.75, msg=f"mode2 RE too large: {best2}")

    def test_helpers(self) -> None:
        """Basic sanity checks for STFT / window helpers."""
        self.assertEqual(nextpow2(5), 3)
        w = gausswin(11)
        self.assertEqual(w.size, 11)
        self.assertAlmostEqual(w[5], 1.0, places=6)

        y = matlab_smooth(np.arange(10, dtype=float), span=3)
        self.assertEqual(y.shape, (10,))
        self.assertTrue(np.all(np.isfinite(y)))

        x = np.cos(2 * np.pi * 10 * np.arange(64) / 64.0)
        sx, sfs = sub_tfrstft(x, length_win=15, fs=64.0)
        self.assertEqual(sx.shape[1], 64)
        self.assertEqual(sfs.size, sx.shape[0])
        self.assertGreater(optwin(x), 0)

    def test_str(self) -> None:
        """Verify string representation."""
        self.assertIn("APMD", str(APMD()))


if __name__ == "__main__":
    unittest.main()
