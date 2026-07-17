# -*- coding: utf-8 -*-
"""
Unit tests for Adaptive Variational Nonlinear Chirp Mode Decomposition (AVNCMD).
"""
import unittest
import numpy as np

from pysdkit import AVNCMD


def _make_chirp_signal(fs: float = 100.0, T: float = 1.0):
    """Two-component nonlinear chirp used by the AVNCMD paper demo (downsampled)."""
    t = np.arange(0.0, T, 1.0 / fs)
    a1 = np.exp(-0.03 * t)
    a2 = np.exp(-0.06 * t)
    f_1t = 25 + 8 * t - 3 * t**2 + 0.4 * t**3
    f_2t = 40 + 16 * t - 6 * t**2 + 0.4 * t**3
    g_1t = a1 * np.cos(2 * np.pi * (0.8 + 25 * t + 4 * t**2 - 1 * t**3 + 0.1 * t**4))
    g_2t = a2 * np.cos(2 * np.pi * (1 + 40 * t + 8 * t**2 - 2 * t**3 + 0.1 * t**4))
    signal = g_1t + g_2t
    iniIF = np.vstack([28.0 * np.ones_like(t), 48.0 * np.ones_like(t)])
    return t, signal, g_1t, g_2t, f_1t, f_2t, iniIF


class AVNCMDTest(unittest.TestCase):
    """Automated tests for AVNCMD."""

    def test_fit_transform_shapes(self) -> None:
        """Verify that fit_transform returns arrays with the expected shapes."""
        t, signal, _, _, _, _, iniIF = _make_chirp_signal(fs=80.0)
        avncmd = AVNCMD(beta=1e-6, tol=1e-4, max_iter=3)
        estIF, estMode, estIA = avncmd.fit_transform(signal, iniIF=iniIF, fs=80.0)

        self.assertEqual(estIF.shape, (2, len(signal)))
        self.assertEqual(estMode.shape, (2, len(signal)))
        self.assertEqual(estIA.shape, (2, len(signal)))

    def test_default_call(self) -> None:
        """Verify that the __call__ interface matches fit_transform."""
        t, signal, _, _, _, _, iniIF = _make_chirp_signal(fs=80.0)
        avncmd = AVNCMD(iniIF=iniIF, fs=80.0, beta=1e-6, tol=1e-4, max_iter=3)
        out_call = avncmd(signal)
        out_fit = avncmd.fit_transform(signal)

        self.assertEqual(len(out_call), 3)
        for a, b in zip(out_call, out_fit):
            self.assertEqual(a.shape, b.shape)

    def test_missing_iniIF_raises(self) -> None:
        """iniIF must be provided either in the constructor or fit_transform."""
        signal = np.random.randn(50)
        avncmd = AVNCMD(max_iter=1)
        with self.assertRaises(ValueError):
            avncmd.fit_transform(signal, fs=50.0)

    def test_history_attributes(self) -> None:
        """After fitting, iteration history attributes should be populated."""
        t, signal, _, _, _, _, iniIF = _make_chirp_signal(fs=80.0)
        avncmd = AVNCMD(beta=1e-6, tol=1e-4, max_iter=3)
        avncmd.fit_transform(signal, iniIF=iniIF, fs=80.0)

        self.assertIsNotNone(avncmd.estIF)
        self.assertIsNotNone(avncmd.estMode)
        self.assertIsNotNone(avncmd.estIA)
        self.assertEqual(avncmd.estIF.ndim, 3)
        self.assertEqual(avncmd.estMode.ndim, 3)
        self.assertEqual(avncmd.estIF.shape[0], 2)
        self.assertEqual(avncmd.estIF.shape[1], len(signal))

    def test_relative_error_bounds(self) -> None:
        """Estimated IF/modes should stay close to the ground-truth chirps."""
        t, signal, g_1t, g_2t, f_1t, f_2t, iniIF = _make_chirp_signal(fs=100.0)
        avncmd = AVNCMD(beta=1e-6, tol=1e-5, max_iter=15)
        estIF, estMode, estIA = avncmd.fit_transform(signal, iniIF=iniIF, fs=100.0)

        self.assertTrue(np.all(np.isfinite(estIF)))
        self.assertTrue(np.all(np.isfinite(estMode)))
        self.assertTrue(np.all(np.isfinite(estIA)))

        # Modes may be returned in either order; pick the better permutation.
        true_modes = [g_1t, g_2t]
        true_ifs = [f_1t, f_2t]
        best = None
        for order in ((0, 1), (1, 0)):
            mode_re = [
                np.linalg.norm(estMode[j] - true_modes[i])
                / np.linalg.norm(true_modes[i])
                for j, i in enumerate(order)
            ]
            if_re = [
                np.linalg.norm(estIF[j] - true_ifs[i]) / np.linalg.norm(true_ifs[i])
                for j, i in enumerate(order)
            ]
            score = sum(mode_re) + sum(if_re)
            if best is None or score < best[0]:
                best = (score, if_re, mode_re)

        _, if_re, mode_re = best
        self.assertLess(if_re[0], 0.2, msg=f"IF1 relative error too large: {if_re[0]}")
        self.assertLess(if_re[1], 0.2, msg=f"IF2 relative error too large: {if_re[1]}")
        self.assertLess(
            mode_re[0], 0.35, msg=f"Mode1 relative error too large: {mode_re[0]}"
        )
        self.assertLess(
            mode_re[1], 0.35, msg=f"Mode2 relative error too large: {mode_re[1]}"
        )

        # Mixed signal should be approximately reconstructed by the sum of modes.
        recon_re = np.linalg.norm(estMode.sum(axis=0) - signal) / np.linalg.norm(signal)
        self.assertLess(
            recon_re, 0.35, msg=f"Reconstruction relative error too large: {recon_re}"
        )

    def test_differ(self) -> None:
        """Verify the discrete derivative helper on a linear ramp."""
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ybar = AVNCMD.differ(y, delta=1.0)
        self.assertEqual(ybar.shape, y.shape)
        self.assertTrue(np.allclose(ybar[1:-1], 1.0))

    def test_str(self) -> None:
        """Verify string representation."""
        self.assertIn("AVNCMD", str(AVNCMD()))


if __name__ == "__main__":
    unittest.main()
