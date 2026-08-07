# -*- coding: utf-8 -*-
"""
Unit tests for Singular Spectral Analysis (SSA).
"""

import unittest

import numpy as np

from pysdkit import SSA
from pysdkit.data import test_emd, test_univariate_signal


def _two_tone_signal(n: int = 200) -> np.ndarray:
    """Synthetic mixture of two pure tones (each tone spans a 2-D SSA subspace)."""
    t = np.arange(n, dtype=float) / n
    return np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 11 * t)


class SSATest(unittest.TestCase):
    """Unit tests for Singular Spectral Analysis (SSA)."""

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition runs and returns a valid shape."""
        ssa = SSA(K=3, mode="traj")
        for index in range(1, 4):
            _, signal = test_univariate_signal(case=index)
            imfs = ssa.fit_transform(signal)

            self.assertEqual(
                len(imfs.shape),
                2,
                msg="The output shape of the decomposed signal is wrong",
            )
            self.assertEqual(
                imfs.shape[0],
                3,
                msg="Number of components should equal K",
            )
            self.assertEqual(
                imfs.shape[1],
                len(signal),
                msg="Wrong length of decomposed signal",
            )
            self.assertTrue(np.all(np.isfinite(imfs)))

    def test_default_call(self) -> None:
        """Verify that __call__ matches fit_transform."""
        _, signal = test_emd()
        ssa = SSA(K=3, mode="traj")

        a = ssa(signal)
        b = ssa.fit_transform(signal)

        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_str(self) -> None:
        """Verify the human-readable algorithm name."""
        self.assertIn("SSA", str(SSA()))
        self.assertIn("Singular Spectral Analysis", str(SSA()))

    def test_reconstruction_two_tone(self) -> None:
        """Four SSA components should nearly reconstruct a two-tone signal."""
        signal = _two_tone_signal(200)
        ssa = SSA(K=4, mode="traj", lags=40)
        imfs = ssa.fit_transform(signal)

        reconstructed = np.sum(imfs, axis=0)
        rel_err = np.linalg.norm(reconstructed - signal) / np.linalg.norm(signal)
        self.assertLess(
            rel_err,
            1e-10,
            msg=f"Two-tone signal should reconstruct well, got rel_err={rel_err}",
        )

    def test_more_components_lower_error(self) -> None:
        """Using more components should not increase reconstruction error."""
        _, signal = test_emd()
        err_small = np.linalg.norm(
            SSA(K=2, mode="traj").fit_transform(signal).sum(0) - signal
        )
        err_large = np.linalg.norm(
            SSA(K=20, mode="traj").fit_transform(signal).sum(0) - signal
        )
        self.assertLessEqual(
            err_large,
            err_small + 1e-8,
            msg="Larger K should yield a better or equal reconstruction",
        )

    def test_modes(self) -> None:
        """Supported lag-matrix modes should produce finite outputs."""
        _, signal = test_emd()
        for mode in ("traj", "trajectory", "caterpillar", "covar", "valid"):
            imfs = SSA(K=2, mode=mode).fit_transform(signal)
            self.assertEqual(imfs.shape[0], 2)
            self.assertEqual(imfs.shape[1], len(signal))
            self.assertTrue(
                np.all(np.isfinite(imfs)),
                msg=f"Non-finite values for mode={mode}",
            )

    def test_custom_lags(self) -> None:
        """Custom window length (lags) should be accepted."""
        signal = _two_tone_signal(160)
        imfs = SSA(K=3, mode="traj", lags=30).fit_transform(signal)
        self.assertEqual(imfs.shape, (3, signal.size))

    def test_averaging_flag(self) -> None:
        """Both averaging and summing diagonal reconstructions should run."""
        signal = _two_tone_signal(120)
        for averaging in (True, False):
            imfs = SSA(K=2, mode="traj", lags=25, averaging=averaging).fit_transform(
                signal
            )
            self.assertEqual(imfs.shape[1], signal.size)
            self.assertTrue(np.all(np.isfinite(imfs)))

    def test_k_capped_by_subspace(self) -> None:
        """K larger than the eigen-subspace size should be safely capped."""
        signal = _two_tone_signal(80)
        lags = 10
        imfs = SSA(K=100, mode="traj", lags=lags).fit_transform(signal)
        self.assertLessEqual(imfs.shape[0], lags)
        self.assertEqual(imfs.shape[1], signal.size)

    def test_invalid_k(self) -> None:
        """K must be a positive integer."""
        with self.assertRaises(ValueError):
            SSA(K=0)

    def test_invalid_lags(self) -> None:
        """lags must satisfy 2 <= lags < len(signal)."""
        signal = np.ones(20)
        with self.assertRaises(ValueError):
            SSA(K=2, lags=1).fit_transform(signal)
        with self.assertRaises(ValueError):
            SSA(K=2, lags=20).fit_transform(signal)

    def test_invalid_signal(self) -> None:
        """Reject empty / non-1D inputs."""
        ssa = SSA(K=2)
        with self.assertRaises(ValueError):
            ssa.fit_transform(np.array([]))
        with self.assertRaises(ValueError):
            ssa.fit_transform(np.ones((5, 5)))


if __name__ == "__main__":
    unittest.main()
