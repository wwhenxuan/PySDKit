# -*- coding: utf-8 -*-
"""
Verify Short-Time Variational Mode Decomposition (STVMD).
"""

import unittest
import numpy as np

from pysdkit import STVMD, stvmd


def _two_tones(n: int = 256, fs: float = 128.0) -> np.ndarray:
    """Stationary two-tone mixture used in the STVMD paper examples."""
    t = np.arange(n) / fs
    return np.sin(2.0 * np.pi * 20.0 * t) + 0.5 * np.sin(2.0 * np.pi * 28.0 * t)


def _freq_hopping(n: int = 512, fs: float = 128.0, seed: int = 7) -> np.ndarray:
    """Piecewise-constant frequency hops (non-stationary)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    seq = rng.permutation(np.arange(8))
    omega = seq[np.floor(t).astype(int) % 8] + 13.0
    x1 = np.sin(2.0 * np.pi * omega * t)
    x2 = 0.5 * np.sin(2.0 * np.pi * (2.0 * omega) * t)
    return x1 + x2 + 0.05 * rng.standard_normal(n)


class STVMDTest(unittest.TestCase):
    """Verify Short-Time Variational Mode Decomposition (STVMD)."""

    signal = _two_tones(256)

    def test_fit_transform_shape(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=80, tol=1e-6)
        modes = decomp.fit_transform(self.signal)
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[0], 3)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_default_call(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=60, tol=1e-6)
        a = decomp(self.signal)
        b = decomp.fit_transform(self.signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_return_all_nodynamic(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=80, tol=1e-6, dynamic=False)
        modes, u_hat, omega = decomp.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape[0], 3)
        self.assertEqual(omega.ndim, 1)
        self.assertEqual(omega.size, 3)
        self.assertEqual(u_hat.shape[2], 3)
        self.assertTrue(np.all(np.diff(omega) >= -1e-12))
        self.assertTrue(np.all(omega >= 0.0) and np.all(omega <= 1.0))

    def test_return_all_dynamic(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=80, tol=1e-6, dynamic=True)
        modes, u_hat, omega = decomp.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape, (3, self.signal.size))
        self.assertEqual(omega.ndim, 2)
        self.assertEqual(omega.shape[0], 3)
        self.assertEqual(omega.shape[1], u_hat.shape[-1])

    def test_functional_interface(self) -> None:
        modes = stvmd(self.signal, K=3, alpha=50.0, n_fft=32, max_iter=60, tol=1e-6)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_reconstruction_nodynamic(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=64, max_iter=200, tol=1e-8, dynamic=False)
        modes = decomp.fit_transform(self.signal)
        recon = np.sum(modes, axis=0)
        rel_err = np.linalg.norm(recon - self.signal) / np.linalg.norm(self.signal)
        self.assertLess(rel_err, 0.15, msg=f"reconstruction RE too large: {rel_err}")

    def test_reconstruction_dynamic(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=64, max_iter=200, tol=1e-8, dynamic=True)
        modes = decomp.fit_transform(self.signal)
        recon = np.sum(modes, axis=0)
        rel_err = np.linalg.norm(recon - self.signal) / np.linalg.norm(self.signal)
        self.assertLess(rel_err, 0.15, msg=f"reconstruction RE too large: {rel_err}")

    def test_multichannel_shape(self) -> None:
        t = np.arange(128) / 128.0
        x1 = np.sin(2.0 * np.pi * 10.0 * t)
        x2 = np.sin(2.0 * np.pi * 20.0 * t)
        multi = np.vstack([x1 + 0.5 * x2, 0.5 * x1 + x2])
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=60, tol=1e-6)
        modes = decomp.fit_transform(multi)
        self.assertEqual(modes.shape, (3, 2, multi.shape[1]))

    def test_dynamic_override(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=50, dynamic=False)
        _, _, omega_nd = decomp.fit_transform(
            self.signal, return_all=True, dynamic=False
        )
        _, _, omega_d = decomp.fit_transform(self.signal, return_all=True, dynamic=True)
        self.assertEqual(omega_nd.ndim, 1)
        self.assertEqual(omega_d.ndim, 2)

    def test_nonstationary_runs(self) -> None:
        sig = _freq_hopping(256)
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=80, dynamic=True)
        modes = decomp.fit_transform(sig)
        self.assertEqual(modes.shape, (3, sig.size))
        self.assertTrue(np.all(np.isfinite(modes)))

    def test_wrong_window(self) -> None:
        with self.assertRaises(ValueError):
            STVMD(window="not-a-window")

    def test_wrong_n_fft(self) -> None:
        with self.assertRaises(ValueError):
            STVMD(n_fft=2)

    def test_str(self) -> None:
        self.assertIn("STVMD", str(STVMD()))

    def test_attributes_after_fit(self) -> None:
        decomp = STVMD(K=3, alpha=50.0, n_fft=32, max_iter=40)
        decomp.fit_transform(self.signal)
        self.assertIsNotNone(decomp.u)
        self.assertIsNotNone(decomp.omega)
        self.assertIsNotNone(decomp.u_hat)
        self.assertIsNotNone(decomp.n_iter)


if __name__ == "__main__":
    unittest.main()
