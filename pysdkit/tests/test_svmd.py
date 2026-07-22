# -*- coding: utf-8 -*-
"""
Created on 2025/07/22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import SVMD, svmd


def _synthetic_tones(n: int = 400) -> np.ndarray:
    """Three well-separated tones used across SVMD tests."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return (
        np.cos(2.0 * np.pi * 3.0 * t)
        + 0.8 * np.cos(2.0 * np.pi * 25.0 * t)
        + 0.5 * np.cos(2.0 * np.pi * 60.0 * t)
    )


class SVMDTest(unittest.TestCase):
    """Verify Successive Variational Mode Decomposition (SVMD)."""

    signal = _synthetic_tones(400)

    def test_fit_transform_shape(self) -> None:
        """Modes should be (L, N) with N matching the (even) input length."""
        decomp = SVMD(max_alpha=5000, stopc=2, max_iter=200, max_modes=8)
        modes = decomp.fit_transform(self.signal)
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.signal.size)
        self.assertGreaterEqual(modes.shape[0], 2)

    def test_default_call(self) -> None:
        """``__call__`` must mirror ``fit_transform``."""
        decomp = SVMD(max_alpha=5000, stopc=2, max_iter=200, max_modes=8)
        a = decomp(self.signal)
        b = decomp.fit_transform(self.signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_return_all(self) -> None:
        """return_all yields modes, spectra and center frequencies."""
        decomp = SVMD(max_alpha=5000, stopc=2, max_iter=200, max_modes=8)
        modes, u_hat, omega = decomp.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape[0], omega.size)
        self.assertEqual(u_hat.shape, (self.signal.size, modes.shape[0]))
        # Center frequencies must be sorted ascending in [0, 0.5]
        self.assertTrue(np.all(np.diff(omega) >= -1e-12))
        self.assertTrue(np.all(omega >= 0.0) and np.all(omega <= 0.5))

    def test_functional_interface(self) -> None:
        """Functional ``svmd`` shortcut."""
        modes = svmd(self.signal, max_alpha=5000, stopc=2, max_iter=200)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_stopc_exact_reconstruction(self) -> None:
        """stopc=2 should recover the three synthetic tones."""
        decomp = SVMD(max_alpha=5000, stopc=2, max_iter=250, max_modes=8)
        modes, _, omega = decomp.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape[0], 3)
        # Normalised frequencies: 3/400, 25/400, 60/400
        expected = np.array([3.0, 25.0, 60.0]) / self.signal.size
        self.assertTrue(
            np.allclose(omega, expected, atol=5e-3),
            msg=f"omega={omega}, expected≈{expected}",
        )

    def test_reconstruction_quality(self) -> None:
        """Sum of modes should approximate the clean input."""
        decomp = SVMD(max_alpha=5000, stopc=2, max_iter=250, max_modes=8)
        modes = decomp.fit_transform(self.signal)
        recon = np.sum(modes, axis=0)
        rel_err = np.linalg.norm(recon - self.signal) / np.linalg.norm(self.signal)
        self.assertLess(rel_err, 0.1)

    def test_odd_length_truncated(self) -> None:
        """Odd-length inputs are truncated by one sample (MATLAB behaviour)."""
        sig = _synthetic_tones(401)
        decomp = SVMD(max_alpha=5000, stopc=2, max_iter=200, max_modes=8)
        modes = decomp.fit_transform(sig)
        self.assertEqual(modes.shape[1], 400)

    def test_stopc_variants(self) -> None:
        """All four stopping criteria should run without error."""
        for stopc in (1, 2, 3, 4):
            decomp = SVMD(
                max_alpha=2000, stopc=stopc, max_iter=150, max_modes=6, tau=0.0
            )
            modes = decomp.fit_transform(self.signal)
            self.assertGreaterEqual(modes.shape[0], 1)
            self.assertEqual(modes.shape[1], self.signal.size)

    def test_wrong_stopc(self) -> None:
        """Invalid stopc must raise at construction time."""
        with self.assertRaises(ValueError):
            SVMD(stopc=9)

    def test_wrong_init_omega(self) -> None:
        """Invalid init_omega must raise at construction time."""
        with self.assertRaises(ValueError):
            SVMD(init_omega=3)

    def test_str(self) -> None:
        """Printable algorithm name."""
        self.assertIn("SVMD", str(SVMD()))

    def test_max_modes_cap(self) -> None:
        """max_modes must bound the number of extracted components."""
        decomp = SVMD(max_alpha=1000, stopc=4, max_iter=80, max_modes=2)
        modes = decomp.fit_transform(self.signal)
        self.assertLessEqual(modes.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
