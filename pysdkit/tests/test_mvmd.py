# -*- coding: utf-8 -*-
"""
Unit tests for Multivariate Variational Mode Decomposition (MVMD).
"""

import unittest

import numpy as np

from pysdkit import MVMD
from pysdkit._vmd.base import Base
from pysdkit._vmd.mvmd import MVMD as ModuleMVMD


def _two_channel_signal(n_samples: int = 256, sampling_rate: float = 256.0):
    """Two-channel mixture matching the MVMD notebook example."""
    time = np.arange(n_samples, dtype=float) / sampling_rate
    channel_1 = np.cos(2.0 * np.pi * 2.0 * time) + np.cos(2.0 * np.pi * 36.0 * time)
    channel_2 = np.cos(2.0 * np.pi * 24.0 * time) + np.cos(2.0 * np.pi * 36.0 * time)
    return np.vstack([channel_1, channel_2]), sampling_rate, time


class MVMDTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.MVMD` and inherited ``Base`` helpers."""

    def setUp(self) -> None:
        self.signal, self.fs, self.time = _two_channel_signal()
        self.n_channels, self.n_samples = self.signal.shape
        self.n_modes = 3
        self.mvmd = MVMD(
            alpha=1000,
            K=self.n_modes,
            tau=0.0,
            init="uniform",
            DC=False,
            tol=1e-6,
            max_iter=40,
        )

    def test_str(self) -> None:
        """``str(MVMD)`` reports the algorithm name."""
        self.assertEqual(
            str(self.mvmd), "Multivariate Variational mode decomposition (MVMD)"
        )

    def test_init_stores_parameters(self) -> None:
        """Constructor stores the requested VMD parameters."""
        self.assertEqual(self.mvmd.alpha, 1000)
        self.assertEqual(self.mvmd.K, 3)
        self.assertEqual(self.mvmd.tau, 0.0)
        self.assertEqual(self.mvmd.init, "uniform")
        self.assertFalse(self.mvmd.DC)
        self.assertEqual(self.mvmd.tol, 1e-6)
        self.assertEqual(self.mvmd.max_iter, 40)
        self.assertEqual(self.mvmd.DTYPE, np.complex64)
        self.assertIsInstance(self.mvmd, Base)

    def test_init_lowercases_init_mode(self) -> None:
        """The ``init`` argument is stored in lowercase."""
        mvmd = MVMD(alpha=1000, K=2, tau=0.0, init="UNIFORM")
        self.assertEqual(mvmd.init, "uniform")

    def test_import_from_package_root(self) -> None:
        """MVMD is exported from the package root and the VMD module."""
        self.assertIs(MVMD, ModuleMVMD)

    def test_fit_transform_shape(self) -> None:
        """``fit_transform`` returns ``K`` real modes of shape ``(K, T, C)``."""
        modes = self.mvmd.fit_transform(self.signal)
        self.assertEqual(modes.shape, (self.n_modes, self.n_samples, self.n_channels))
        self.assertFalse(np.iscomplexobj(modes))
        self.assertTrue(np.all(np.isfinite(modes)))

    def test_fit_transform_return_all(self) -> None:
        """``return_all=True`` also returns spectra and center frequencies."""
        modes, spectra, omega = self.mvmd.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape, (self.n_modes, self.n_samples, self.n_channels))
        self.assertEqual(spectra.shape, (self.n_samples, self.n_modes, self.n_channels))
        self.assertEqual(omega.ndim, 2)
        self.assertEqual(omega.shape[1], self.n_modes)
        self.assertGreaterEqual(omega.shape[0], 1)
        self.assertLessEqual(omega.shape[0], self.mvmd.max_iter)
        self.assertTrue(np.iscomplexobj(spectra))
        np.testing.assert_allclose(modes, np.real(modes))

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        modes_fit = self.mvmd.fit_transform(self.signal, return_all=False)
        modes_call = self.mvmd(self.signal, return_all=False)
        np.testing.assert_allclose(modes_call, modes_fit)

    def test_call_return_all(self) -> None:
        """``__call__`` forwards ``return_all`` to ``fit_transform``."""
        result = self.mvmd(self.signal, return_all=True)
        self.assertEqual(len(result), 3)

    def test_init_omega_uniform(self) -> None:
        """Uniform initialization spaces center frequencies in ``[0, 0.5)``."""
        omega = self.mvmd._MVMD__init_omega(fs=1.0)
        self.assertEqual(omega.shape, (self.mvmd.max_iter, self.n_modes))
        expected = np.array([(0.5 / self.n_modes) * i for i in range(self.n_modes)])
        np.testing.assert_allclose(np.real(omega[0]), expected)
        np.testing.assert_array_equal(omega[1:], 0)

    def test_init_omega_zero(self) -> None:
        """``init='zero'`` starts every mode at frequency 0."""
        mvmd = MVMD(alpha=1000, K=4, tau=0.0, init="zero", max_iter=5)
        omega = mvmd._MVMD__init_omega(fs=1.0)
        np.testing.assert_array_equal(omega, np.zeros((5, 4)))

    def test_init_omega_random_is_sorted(self) -> None:
        """``init='random'`` draws sorted, finite initial frequencies."""
        np.random.seed(0)
        mvmd = MVMD(alpha=1000, K=4, tau=0.0, init="random", max_iter=5)
        omega = mvmd._MVMD__init_omega(fs=1.0)
        self.assertEqual(omega.shape, (5, 4))
        self.assertTrue(np.all(np.diff(np.real(omega[0])) >= 0))
        self.assertTrue(np.all(np.isfinite(omega[0])))

    def test_init_omega_dc_forces_first_mode_to_zero(self) -> None:
        """``DC=True`` keeps the first mode at zero frequency."""
        mvmd = MVMD(alpha=1000, K=3, tau=0.0, init="uniform", DC=True, max_iter=5)
        omega = mvmd._MVMD__init_omega(fs=1.0)
        self.assertEqual(omega[0, 0], 0.0)

    def test_fit_transform_init_zero(self) -> None:
        """Decomposition succeeds with zero frequency initialization."""
        mvmd = MVMD(alpha=1000, K=2, tau=0.0, init="zero", max_iter=20)
        modes = mvmd.fit_transform(self.signal)
        self.assertEqual(modes.shape[0], 2)
        self.assertEqual(modes.shape[1], self.n_samples)

    def test_fit_transform_init_random(self) -> None:
        """Decomposition succeeds with random frequency initialization."""
        np.random.seed(1)
        mvmd = MVMD(alpha=1000, K=2, tau=0.0, init="random", max_iter=20)
        modes = mvmd.fit_transform(self.signal)
        self.assertEqual(modes.shape, (2, self.n_samples, self.n_channels))

    def test_fit_transform_dc_true(self) -> None:
        """``DC=True`` still returns finite modes of the requested rank."""
        mvmd = MVMD(alpha=1000, K=3, tau=0.0, init="uniform", DC=True, max_iter=20)
        modes, _, omega = mvmd.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape[0], 3)
        self.assertEqual(omega[0, 0], 0.0)
        self.assertTrue(np.all(np.isfinite(modes)))

    def test_reconstruction_error_is_finite(self) -> None:
        """Summing modes over ``K`` reconstructs a finite multivariate signal."""
        modes = self.mvmd.fit_transform(self.signal)
        reconstructed = np.sum(modes, axis=0)
        self.assertEqual(reconstructed.shape, (self.n_samples, self.n_channels))
        self.assertTrue(np.all(np.isfinite(reconstructed)))
        self.assertGreater(np.linalg.norm(reconstructed), 0.0)

    def test_fft_matches_numpy(self) -> None:
        """Inherited ``fft`` matches :func:`numpy.fft.fft`."""
        values = np.arange(8, dtype=float)
        np.testing.assert_allclose(self.mvmd.fft(values), np.fft.fft(values))

    def test_ifft_matches_numpy(self) -> None:
        """Inherited ``ifft`` matches :func:`numpy.fft.ifft`."""
        values = np.arange(8, dtype=float) + 1j * np.arange(8, dtype=float)[::-1]
        np.testing.assert_allclose(self.mvmd.ifft(values), np.fft.ifft(values))

    def test_fftshift_matches_numpy(self) -> None:
        """Inherited ``fftshift`` matches :func:`numpy.fft.fftshift`."""
        values = np.arange(9, dtype=float)
        np.testing.assert_array_equal(
            self.mvmd.fftshift(values), np.fft.fftshift(values)
        )

    def test_ifftshift_matches_numpy(self) -> None:
        """Inherited ``ifftshift`` matches :func:`numpy.fft.ifftshift`."""
        values = np.arange(9, dtype=float)
        np.testing.assert_array_equal(
            self.mvmd.ifftshift(values), np.fft.ifftshift(values)
        )

    def test_fmirror(self) -> None:
        """``fmirror`` prepends and appends ``sym`` mirrored samples."""
        series = np.arange(6, dtype=float)
        mirrored = self.mvmd.fmirror(series, sym=2)
        self.assertEqual(mirrored.size, series.size + 4)
        np.testing.assert_array_equal(mirrored[:2], np.flip(series[:2]))
        np.testing.assert_array_equal(mirrored[2:8], series)
        np.testing.assert_array_equal(mirrored[-2:], np.flip(series[-2:]))

    def test_multi_fmirror(self) -> None:
        """``multi_fmirror`` doubles each channel by odd-symmetric extension."""
        mirrored = self.mvmd.multi_fmirror(
            self.signal, C=self.n_channels, T=self.n_samples
        )
        self.assertEqual(mirrored.shape, (self.n_channels, 2 * self.n_samples))
        np.testing.assert_array_equal(
            mirrored[:, self.n_samples // 2 : self.n_samples // 2 + self.n_samples],
            self.signal,
        )

    def test_fft_roundtrip(self) -> None:
        """``ifft(fft(x))`` recovers the original 1-D series."""
        values = np.linspace(-1.0, 1.0, 16)
        recovered = self.mvmd.ifft(self.mvmd.fft(values))
        np.testing.assert_allclose(recovered.real, values, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
