# -*- coding: utf-8 -*-
"""
Unit tests for Multivariate Empirical Mode Decomposition (MEMD).
"""

import unittest

import numpy as np

from pysdkit import MEMD
from pysdkit._emd.memd import MEMD as ModuleMEMD
from pysdkit._emd.memd import (
    boundary_conditions,
    envelope_mean,
    hamm,
    is_prime,
    local_peaks,
    nth_prime,
    peaks,
    spherical_coordinate_directions,
    unit_direction,
    zero_crossings,
)
from pysdkit.data import (
    load_memd_syn_12channel,
    load_memd_syn_16channel,
    load_memd_syn_hex,
    load_memd_taichi_hex,
)


def _trivariate_tones(n_samples: int = 256, sampling_rate: float = 256.0) -> np.ndarray:
    """Three-channel mixture with a shared 4 Hz tone and extra harmonics."""
    time = np.arange(n_samples, dtype=float) / sampling_rate
    channel_0 = np.cos(2.0 * np.pi * 4.0 * time) + 0.5 * np.cos(
        2.0 * np.pi * 18.0 * time
    )
    channel_1 = np.cos(2.0 * np.pi * 4.0 * time) + 0.4 * np.sin(
        2.0 * np.pi * 11.0 * time
    )
    channel_2 = 0.8 * np.cos(2.0 * np.pi * 18.0 * time) + 0.3 * np.cos(
        2.0 * np.pi * 4.0 * time
    )
    return np.vstack([channel_0, channel_1, channel_2])


class MEMDTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.MEMD` and module-level helpers."""

    def setUp(self) -> None:
        self.signal = _trivariate_tones()
        self.n_channels, self.n_samples = self.signal.shape
        self.memd = MEMD(n_dir=8, max_iter=40, stop_cnt=2)

    def test_str(self) -> None:
        """``str(MEMD)`` reports the algorithm name."""
        self.assertEqual(
            str(self.memd), "Multivariate Empirical Mode Decomposition (MEMD)"
        )

    def test_init_stores_parameters(self) -> None:
        """Constructor stores stopping and projection parameters."""
        memd = MEMD(
            stop_crit="fix_h",
            max_iter=50,
            n_dir=12,
            stop_vec=[0.05, 0.5, 0.05],
            stop_cnt=3,
        )
        self.assertEqual(memd.stop_crit, "fix_h")
        self.assertEqual(memd.max_iter, 50)
        self.assertEqual(memd.n_dir, 12)
        self.assertEqual(memd.stop_cnt, 3)
        np.testing.assert_allclose(memd.stop_vec, [0.05, 0.5, 0.05])
        self.assertEqual(memd.sd, 0.05)
        self.assertEqual(memd.sd2, 0.5)
        self.assertEqual(memd.tol, 0.05)

    def test_import_from_package_root(self) -> None:
        """MEMD is exported from the package root and the ``_emd`` module."""
        self.assertIs(MEMD, ModuleMEMD)

    def test_invalid_stop_crit(self) -> None:
        """Unknown stopping criteria raise ``ValueError``."""
        with self.assertRaises(ValueError):
            MEMD(stop_crit="unknown")

    def test_invalid_n_dir(self) -> None:
        """Fewer than 6 direction vectors is rejected."""
        with self.assertRaises(ValueError):
            MEMD(n_dir=5)

    def test_invalid_stop_vec(self) -> None:
        """``stop_vec`` must contain exactly three finite numbers."""
        with self.assertRaises(ValueError):
            MEMD(stop_vec=[0.1, 0.2])

    def test_invalid_stop_cnt(self) -> None:
        """Negative ``stop_cnt`` is rejected."""
        with self.assertRaises(ValueError):
            MEMD(stop_cnt=-1)

    def test_invalid_max_iter(self) -> None:
        """Non-positive ``max_iter`` is rejected."""
        with self.assertRaises(ValueError):
            MEMD(max_iter=0)

    def test_fit_transform_shape(self) -> None:
        """``fit_transform`` returns ``(K, T, C)`` with finite values."""
        imfs = self.memd.fit_transform(self.signal)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1], self.n_samples)
        self.assertEqual(imfs.shape[2], self.n_channels)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        """Summing IMFs including the residue recovers the input."""
        imfs = self.memd.fit_transform(self.signal)
        reconstructed = imfs.sum(axis=0).T
        np.testing.assert_allclose(reconstructed, self.signal, atol=1e-8)

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        a = self.memd(self.signal)
        b = self.memd.fit_transform(self.signal)
        np.testing.assert_allclose(a, b)

    def test_n_dir_not_overwritten(self) -> None:
        """``n_dir`` stays at the constructor value after a decomposition."""
        memd = MEMD(n_dir=16, max_iter=20)
        memd.fit_transform(self.signal)
        self.assertEqual(memd.n_dir, 16)

    def test_fit_transform_fix_h(self) -> None:
        """``stop_crit='fix_h'`` also yields a reconstructing decomposition."""
        imfs = MEMD(stop_crit="fix_h", n_dir=8, max_iter=30, stop_cnt=2).fit_transform(
            self.signal
        )
        self.assertEqual(imfs.shape[1:], (self.n_samples, self.n_channels))
        np.testing.assert_allclose(imfs.sum(axis=0).T, self.signal, atol=1e-8)

    def test_four_channel_input(self) -> None:
        """Hex-less 4-channel data uses the n-sphere Hammersley map."""
        t = np.arange(180, dtype=float) / 180.0
        signal = np.vstack(
            [
                np.cos(2.0 * np.pi * 3.0 * t),
                np.cos(2.0 * np.pi * 7.0 * t),
                np.sin(2.0 * np.pi * 3.0 * t),
                np.sin(2.0 * np.pi * 11.0 * t),
            ]
        )
        imfs = MEMD(n_dir=8, max_iter=25).fit_transform(signal)
        self.assertEqual(imfs.shape[1:], (signal.shape[1], 4))
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-8)

    def test_transposes_time_major_input(self) -> None:
        """A MATLAB-style ``(n_samples, n_channels)`` array is accepted."""
        imfs = self.memd.fit_transform(self.signal.T)
        self.assertEqual(imfs.shape[1], self.n_samples)
        self.assertEqual(imfs.shape[2], self.n_channels)
        np.testing.assert_allclose(imfs.sum(axis=0), self.signal.T, atol=1e-8)

    def test_rejects_too_few_channels(self) -> None:
        """Univariate / bivariate inputs raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.memd.fit_transform(np.ones((2, 64)))

    def test_rejects_too_many_channels(self) -> None:
        """More than 16 channels is outside the MATLAB toolbox limit."""
        with self.assertRaises(ValueError):
            self.memd.fit_transform(np.ones((17, 64)))

    def test_rejects_1d_and_short_series(self) -> None:
        """Non-2-D arrays and very short records are rejected."""
        with self.assertRaises(ValueError):
            self.memd.fit_transform(np.ones(32))
        with self.assertRaises(ValueError):
            self.memd.fit_transform(np.ones((3, 2)))

    def test_init_hammersley_trivariate(self) -> None:
        """Trivariate Hammersley sequence has two coordinates per direction."""
        seq = self.memd.init_hammersley(3)
        self.assertEqual(seq.shape, (self.memd.n_dir, 2))
        self.assertTrue(np.all(np.isfinite(seq)))
        self.assertTrue(np.all(seq >= 0.0))

    def test_init_hammersley_higher_dim(self) -> None:
        """For n>3 the sequence has one Hammersley coordinate per dimension."""
        seq = MEMD(n_dir=8).init_hammersley(5)
        self.assertEqual(seq.shape, (8, 5))

    def test_direction_vectors_unit_length(self) -> None:
        """Each Hammersley direction is a unit vector."""
        for n_dim in (3, 4, 6):
            dirs = MEMD(n_dir=8).direction_vectors(n_dim)
            self.assertEqual(dirs.shape, (8, n_dim))
            np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-10)

    def test_stop_emd_on_constant(self) -> None:
        """A constant residual has too few extrema on every projection."""
        seq = self.memd.init_hammersley(3)
        constant = np.ones((40, 3))
        self.assertTrue(self.memd.stop_emd(constant, seq, 3))

    def test_stop_emd_on_oscillation(self) -> None:
        """An oscillating residual still has extrema, so sifting continues."""
        seq = self.memd.init_hammersley(3)
        oscillating = self.signal.T
        self.assertFalse(self.memd.stop_emd(oscillating, seq, 3))

    def test_stop_returns_envelope(self) -> None:
        """``stop`` returns a boolean flag and an envelope of shape ``(T, C)``."""
        seq = self.memd.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        flag, env = self.memd.stop(
            self.signal.T, time, seq, self.n_samples, self.n_channels
        )
        self.assertIsInstance(flag, (bool, np.bool_))
        self.assertEqual(env.shape, (self.n_samples, self.n_channels))
        self.assertTrue(np.all(np.isfinite(env)))

    def test_fix_counter(self) -> None:
        """``fix`` increments or resets the consecutive-sifting counter."""
        seq = self.memd.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        flag, env, counter = self.memd.fix(
            self.signal.T, time, seq, self.n_samples, self.n_channels, counter=0
        )
        self.assertIsInstance(flag, (bool, np.bool_))
        self.assertEqual(env.shape, (self.n_samples, self.n_channels))
        self.assertIsInstance(counter, int)
        self.assertGreaterEqual(counter, 0)

    def test_is_prime(self) -> None:
        """Primality matches the list used by MATLAB ``memd.m``."""
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(1))
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(9))
        self.assertTrue(is_prime(97))
        self.assertFalse(is_prime(100))

    def test_nth_prime(self) -> None:
        """``nth_prime`` returns the first n primes, including 3 not 4."""
        self.assertEqual(nth_prime(1), [2])
        self.assertEqual(nth_prime(5), [2, 3, 5, 7, 11])
        self.assertEqual(nth_prime(0), [])

    def test_hamm_negative_base(self) -> None:
        """Negative Hammersley base uses the MATLAB ``(mod(i, n+1)+0.5)/n`` rule."""
        seq = hamm(8, -8)
        self.assertEqual(seq.shape, (8,))
        expected = (np.remainder(np.arange(1, 9), 9) + 0.5) / 8.0
        np.testing.assert_allclose(seq, expected)

    def test_hamm_prime_base(self) -> None:
        """Prime-base van der Corput samples lie in (0, 1)."""
        seq = hamm(16, 2)
        self.assertEqual(seq.shape, (16,))
        self.assertTrue(np.all(seq > 0.0) and np.all(seq < 1.0))

    def test_peaks_and_local_peaks(self) -> None:
        """Sine-wave extrema are recovered by ``peaks`` and ``local_peaks``."""
        time = np.arange(200, dtype=float)
        x = np.sin(2.0 * np.pi * time / 20.0)
        pks, locs = peaks(x)
        self.assertGreater(pks.size, 0)
        np.testing.assert_array_equal(pks, x[locs])
        self.assertTrue(np.all(pks > 0.5))

        indmin, indmax = local_peaks(x)
        self.assertGreater(indmin.size, 2)
        self.assertGreater(indmax.size, 2)
        self.assertTrue(np.all(x[indmax] > 0.5))
        self.assertTrue(np.all(x[indmin] < -0.5))

    def test_local_peaks_flat_signal(self) -> None:
        """A near-zero / constant projection has no extrema."""
        indmin, indmax = local_peaks(np.zeros(50))
        self.assertEqual(indmin.size, 0)
        self.assertEqual(indmax.size, 0)

    def test_zero_crossings(self) -> None:
        """Sign changes and exact zeros are reported."""
        x = np.array([1.0, -1.0, -0.5, 0.2, 0.0, 0.3])
        zc = zero_crossings(x)
        self.assertTrue(np.any(zc == 0))
        self.assertTrue(np.any(zc == 4))

    def test_zero_crossings_zero_run(self) -> None:
        """A run of exact zeros contributes a midpoint index."""
        x = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
        zc = zero_crossings(x)
        self.assertGreaterEqual(zc.size, 1)

    def test_boundary_conditions_mode(self) -> None:
        """Mirror extension is enabled only when there are enough extrema."""
        time = np.arange(1, 101, dtype=float)
        x = np.sin(2.0 * np.pi * time / 20.0)
        z = np.column_stack([x, np.cos(2.0 * np.pi * time / 20.0), 0.5 * x])
        indmin, indmax = local_peaks(x)
        tmin, tmax, zmin, zmax, mode = boundary_conditions(
            indmin, indmax, time, x, z, 2
        )
        self.assertEqual(mode, 1)
        self.assertEqual(zmin.shape[1], 3)
        self.assertEqual(zmax.shape[1], 3)
        self.assertGreater(tmin.size, indmin.size)
        self.assertTrue(np.all(np.diff(tmin) > 0))
        self.assertTrue(np.all(np.diff(tmax) > 0))

        none_vals = boundary_conditions(np.array([2]), np.array([8]), time, x, z, 2)
        self.assertEqual(none_vals[-1], 0)
        self.assertTrue(all(v is None for v in none_vals[:-1]))

    def test_envelope_mean_shape(self) -> None:
        """``envelope_mean`` returns finite envelopes and extrema counts."""
        seq = self.memd.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        env, nem, nzm, amp = envelope_mean(
            self.signal.T, time, seq, self.memd.n_dir, self.n_samples, 3
        )
        self.assertEqual(env.shape, (self.n_samples, 3))
        self.assertEqual(nem.shape, (self.memd.n_dir,))
        self.assertEqual(nzm.shape, (self.memd.n_dir,))
        self.assertEqual(amp.shape, (self.n_samples,))
        self.assertTrue(np.all(np.isfinite(env)))
        self.assertTrue(np.all(nem >= 0))

    def test_unit_direction_trivariate(self) -> None:
        """The 2-sphere map of a Hammersley sample is a unit vector."""
        seq = MEMD(n_dir=8).init_hammersley(3)
        vec = unit_direction(seq, 0, 3)
        self.assertEqual(vec.shape, (3,))
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-12)

    def test_spherical_coordinate_directions(self) -> None:
        """Angular grids cluster near the poles relative to Hammersley."""
        grid = spherical_coordinate_directions(n_phi=12, n_theta=6)
        self.assertEqual(grid.shape, (72, 3))
        np.testing.assert_allclose(np.linalg.norm(grid, axis=1), 1.0, atol=1e-10)
        hammersley = MEMD(n_dir=64).direction_vectors(3)
        pole_axis = np.abs(grid[:, 0])
        self.assertGreater(pole_axis.max(), 0.98)
        self.assertTrue(np.any(np.abs(hammersley[:, 2]) < 0.2))

    def test_packaged_hex_loader(self) -> None:
        """Packaged MATLAB hex / 12 / 16 / Tai-chi arrays keep channel-major layout."""
        hex_data = load_memd_syn_hex()
        self.assertEqual(hex_data["signal"].shape, (6, 1001))
        self.assertEqual(hex_data["t"].shape, (1001,))

        ch12 = load_memd_syn_12channel()
        self.assertEqual(ch12["signal"].shape, (12, 1001))

        ch16 = load_memd_syn_16channel()
        self.assertEqual(ch16["signal"].shape, (16, 1001))

        taichi = load_memd_taichi_hex()
        self.assertEqual(taichi["signal"].shape, (6, 800))
        self.assertTrue(np.all(np.isfinite(taichi["signal"])))

    def test_packaged_hex_smoke(self) -> None:
        """A short slice of the MATLAB hex demo reconstructs after MEMD."""
        signal = load_memd_syn_hex()["signal"][:, :160]
        imfs = MEMD(n_dir=8, max_iter=20).fit_transform(signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
