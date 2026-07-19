# -*- coding: utf-8 -*-
"""
Unit tests for Orthogonalized Variational Mode Decomposition (OVMD).
"""
import unittest
import numpy as np

from pysdkit import OVMD


def _make_multitone(fs: float = 100.0, T: float = 4.0, seed: int = 0):
    """Four stationary tones similar to a shortened OVMD example."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, T, 1.0 / fs)
    comps = [
        0.75 * np.cos(2 * np.pi * 0.1 * t),
        0.65 * np.cos(2 * np.pi * 1.0 * t),
        1.00 * np.cos(2 * np.pi * 20.0 * t),
        0.35 * np.cos(2 * np.pi * 40.0 * t),
    ]
    signal = np.sum(comps, axis=0) + 0.02 * rng.standard_normal(t.size)
    return t, signal, comps


class OVMDTest(unittest.TestCase):
    """Automated tests for OVMD."""

    def test_fit_transform_shapes(self) -> None:
        t, signal, _ = _make_multitone()
        ovmd = OVMD(alpha=100.0, K=5, tol=1e-4, max_iter=80)
        u = ovmd.fit_transform(signal)

        self.assertEqual(u.shape[0], 5)
        self.assertEqual(u.shape[1], signal.size - (signal.size % 2))
        self.assertIsNotNone(ovmd.omega)
        self.assertEqual(ovmd.omega.shape[1], 5)

    def test_return_all(self) -> None:
        t, signal, _ = _make_multitone()
        ovmd = OVMD(alpha=100.0, K=4, tol=1e-4, max_iter=60)
        u, u_hat, omega = ovmd.fit_transform(signal, return_all=True)

        self.assertEqual(u.shape[0], 4)
        self.assertEqual(u_hat.shape[1], 4)
        self.assertEqual(u_hat.shape[0], u.shape[1])
        self.assertEqual(omega.shape[1], 4)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(omega)))

    def test_call_matches_fit_transform(self) -> None:
        t, signal, _ = _make_multitone(T=3.0)
        ovmd = OVMD(alpha=100.0, K=4, tol=1e-4, max_iter=50)
        a = ovmd(signal)
        b = ovmd.fit_transform(signal)
        self.assertTrue(np.allclose(a, b))

    def test_reconstruction_error(self) -> None:
        t, signal, _ = _make_multitone(T=5.0)
        if signal.size % 2:
            signal = signal[:-1]
        ovmd = OVMD(alpha=100.0, K=6, tol=1e-5, max_iter=200)
        u = ovmd.fit_transform(signal)
        recon_re = np.linalg.norm(u.sum(axis=0) - signal) / np.linalg.norm(signal)
        self.assertLess(recon_re, 0.15, msg=f"reconstruction RE too large: {recon_re}")

    def test_modes_nearly_orthogonal(self) -> None:
        """Orthogonality is the key OVMD contribution; off-diagonal corr should be small."""
        t, signal, _ = _make_multitone(T=5.0)
        if signal.size % 2:
            signal = signal[:-1]
        ovmd = OVMD(alpha=100.0, K=6, tol=1e-5, max_iter=200)
        u = ovmd.fit_transform(signal)
        uc = u - u.mean(axis=1, keepdims=True)
        corr = np.corrcoef(uc)
        off = corr - np.eye(corr.shape[0])
        self.assertLess(
            np.max(np.abs(off)),
            0.25,
            msg=f"modes not sufficiently orthogonal: {np.max(np.abs(off))}",
        )

    def test_center_frequencies_ordered_energy(self) -> None:
        """Returned modes are sorted by spectral energy (descending)."""
        t, signal, _ = _make_multitone(T=4.0)
        ovmd = OVMD(alpha=100.0, K=4, tol=1e-4, max_iter=80)
        u, u_hat, _ = ovmd.fit_transform(signal, return_all=True)
        energy = np.sum(np.abs(u_hat) ** 2, axis=0)
        self.assertTrue(np.all(np.diff(energy) <= 1e-8))

    def test_invalid_k_raises(self) -> None:
        ovmd = OVMD(K=0)
        with self.assertRaises(ValueError):
            ovmd.fit_transform(np.random.randn(64))

    def test_str(self) -> None:
        self.assertIn("OVMD", str(OVMD()))


if __name__ == "__main__":
    unittest.main()
