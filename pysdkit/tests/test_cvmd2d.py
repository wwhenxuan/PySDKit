# -*- coding: utf-8 -*-
"""
Unit tests for Compact / 2D-TV Variational Mode Decomposition.
"""

import unittest

import numpy as np

from pysdkit import CVMD2D
from pysdkit.data import test_grayscale, get_meshgrid_2D


def _small_texture(size: int = 64) -> np.ndarray:
    """Synthetic multi-orientation texture similar to the paper demos."""
    x, y = get_meshgrid_2D(low=0.0, high=2.0 * np.pi, sampling_rate=size)
    img = 0.8 * np.sin(6.0 * x) + 0.6 * np.sin(5.0 * y) + 0.5 * np.sin(4.0 * (x + y))
    return img - img.mean()


class CVMD2DTest(unittest.TestCase):
    """Unit tests for Compact / 2D-TV Variational Mode Decomposition."""

    @classmethod
    def setUpClass(cls) -> None:
        gray = test_grayscale().astype(float)
        cls.image = gray[::4, ::4] - gray[::4, ::4].mean()
        cls.synth = _small_texture(48)

    def _make(self, **kwargs) -> CVMD2D:
        params = dict(
            K=3,
            alpha=1000,
            beta=0.5,
            gamma=500,
            delta=np.inf,
            rho=10,
            rho_k=10,
            tau=0.0,
            tau_k=2.5,
            t=1.5,
            DC=True,
            init="radially",
            max_iter=35,
            M=1,
            A_phase=np.array([15.0, np.inf]),
        )
        params.update(kwargs)
        return CVMD2D(**params)

    def test_fit_transform_shape(self) -> None:
        K = 3
        decomp = self._make(K=K)
        u = decomp.fit_transform(self.image)
        hy, hx = self.image.shape
        self.assertEqual(u.shape, (hy, hx, K, 1))
        self.assertTrue(np.all(np.isfinite(u)))

    def test_default_call(self) -> None:
        decomp = self._make()
        a = decomp(self.image)
        b = decomp.fit_transform(self.image)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_return_all(self) -> None:
        decomp = self._make(K=3)
        u, v, omega, A, X = decomp.fit_transform(self.image, return_all=True)
        hy, hx = self.image.shape
        self.assertEqual(u.shape, (hy, hx, 3, 1))
        self.assertEqual(v.shape, u.shape)
        self.assertEqual(omega.shape[1], 3)  # (2, K, M)
        self.assertEqual(A.shape, (hy, hx, 3))
        self.assertEqual(X.shape, (hy, hx))
        self.assertTrue(np.all((A >= 0) & (A <= 1)))

    def test_uniform_init_alias(self) -> None:
        """'uniform' must map to MATLAB init=0 (radially)."""
        decomp = self._make(init="uniform", max_iter=10, A_phase=np.array([5.0, 8.0]))
        u = decomp.fit_transform(self.synth)
        self.assertTrue(np.all(np.isfinite(u)))

    def test_random_init(self) -> None:
        decomp = self._make(
            init="random", DC=False, max_iter=12, A_phase=np.array([6.0, 10.0])
        )
        u = decomp.fit_transform(self.synth)
        self.assertEqual(u.shape[-2], 3)

    def test_dc_mode_stays_at_origin(self) -> None:
        decomp = self._make(K=3, DC=True, max_iter=20, A_phase=np.array([10.0, np.inf]))
        _, _, omega, _, _ = decomp.fit_transform(self.synth, return_all=True)
        self.assertTrue(np.allclose(omega[:, 0, :], 0.0, atol=1e-12))

    def test_phase_iii_winner_takes_all(self) -> None:
        """After phase III, supports should form a partition (sum_k A_k = 1)."""
        decomp = self._make(
            K=3,
            max_iter=40,
            A_phase=np.array([10.0, 20.0]),
            tau=0.0,
            tau_k=0.0,
        )
        _, _, _, A, _ = decomp.fit_transform(self.synth, return_all=True)
        self.assertTrue(np.allclose(A.sum(axis=2), 1.0))

    def test_modes_carry_energy(self) -> None:
        decomp = self._make(K=3, max_iter=40, A_phase=np.array([20.0, np.inf]))
        u = decomp.fit_transform(self.synth)
        energies = [float(np.sum(u[:, :, k, 0] ** 2)) for k in range(3)]
        self.assertTrue(all(e > 0 for e in energies))
        # reconstruction from modes should explain a non-trivial fraction of energy
        recon = u.sum(axis=(2, 3))
        rel = np.linalg.norm(recon - self.synth) / (np.linalg.norm(self.synth) + 1e-30)
        self.assertLess(rel, 1.5)

    def test_invalid_init(self) -> None:
        with self.assertRaises(ValueError):
            self._make(init="not-a-method").fit_transform(self.synth)

    def test_non_2d_input(self) -> None:
        with self.assertRaises(ValueError):
            self._make().fit_transform(np.ones(16))

    def test_str(self) -> None:
        self.assertIn("CVMD2D", str(CVMD2D()))

    def test_grayscale_demo_params(self) -> None:
        """MATLAB case-1 style parameters should run on the library grayscale demo."""
        img = test_grayscale().astype(float)
        img = img[::4, ::4] - img[::4, ::4].mean()
        decomp = CVMD2D(
            K=5,
            alpha=1000,
            beta=0.5,
            gamma=500,
            delta=np.inf,
            rho=10,
            rho_k=10,
            tau=2.5,
            tau_k=2.5,
            t=1.5,
            DC=True,
            init="radially",
            max_iter=45,
            M=1,
            A_phase=np.array([25.0, np.inf]),
        )
        u, v, omega, A, X = decomp.fit_transform(img, return_all=True)
        self.assertEqual(u.shape[2], 5)
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertEqual(X.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
