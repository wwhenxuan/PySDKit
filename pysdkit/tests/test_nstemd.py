# -*- coding: utf-8 -*-
"""
Unit tests for Nonuniformly Sampled Trivariate EMD (NS-TEMD).
"""

import unittest

import numpy as np

from pysdkit import NSTEMD
from pysdkit._emd.nstemd import NSTEMD as ModuleNSTEMD
from pysdkit._emd.nstemd import (
    ellipsoid_directions,
    envelope_mean,
    princomp,
    projection_directions,
)
from pysdkit._emd.memd import local_peaks


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


def _power_imbalanced(n_samples: int = 220) -> np.ndarray:
    """Trivariate record whose energy is concentrated on one axis."""
    time = np.arange(n_samples, dtype=float) / float(n_samples)
    strong = np.cos(2.0 * np.pi * 6.0 * time) + 0.3 * np.cos(2.0 * np.pi * 19.0 * time)
    weak = 0.05 * np.sin(2.0 * np.pi * 6.0 * time)
    weaker = 0.02 * np.cos(2.0 * np.pi * 11.0 * time)
    return np.vstack([10.0 * strong, weak, weaker])


class NSTEMDTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.NSTEMD` and module-level helpers."""

    def setUp(self) -> None:
        self.signal = _trivariate_tones()
        self.n_channels, self.n_samples = self.signal.shape
        self.nstemd = NSTEMD(n_dir=8, max_iter=40, stop_cnt=2)

    def test_str(self) -> None:
        """``str(NSTEMD)`` reports the algorithm name."""
        self.assertEqual(
            str(self.nstemd),
            "Nonuniformly Sampled Trivariate Empirical Mode Decomposition " "(NS-TEMD)",
        )

    def test_init_stores_parameters(self) -> None:
        """Constructor stores stopping and projection parameters."""
        nstemd = NSTEMD(
            stop_crit="fix_h",
            max_iter=50,
            n_dir=12,
            stop_vec=[0.05, 0.5, 0.05],
            stop_cnt=3,
        )
        self.assertEqual(nstemd.stop_crit, "fix_h")
        self.assertEqual(nstemd.max_iter, 50)
        self.assertEqual(nstemd.n_dir, 12)
        self.assertEqual(nstemd.stop_cnt, 3)
        np.testing.assert_allclose(nstemd.stop_vec, [0.05, 0.5, 0.05])

    def test_import_from_package_root(self) -> None:
        """NSTEMD is exported from the package root and the ``_emd`` module."""
        self.assertIs(NSTEMD, ModuleNSTEMD)

    def test_invalid_n_dir(self) -> None:
        """Fewer than 6 direction vectors is rejected."""
        with self.assertRaises(ValueError):
            NSTEMD(n_dir=5)

    def test_invalid_stop_crit(self) -> None:
        """Unknown stopping criteria raise ``ValueError``."""
        with self.assertRaises(ValueError):
            NSTEMD(stop_crit="unknown")

    def test_fit_transform_shape(self) -> None:
        """``fit_transform`` returns ``(K, T, C)`` with finite values."""
        imfs = self.nstemd.fit_transform(self.signal)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1], self.n_samples)
        self.assertEqual(imfs.shape[2], self.n_channels)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        """Summing IMFs including the residue recovers the input."""
        imfs = self.nstemd.fit_transform(self.signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, self.signal, atol=1e-8)

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        a = self.nstemd(self.signal)
        b = self.nstemd.fit_transform(self.signal)
        np.testing.assert_allclose(a, b)

    def test_pca_stored_from_original(self) -> None:
        """PCA of the original record is stored after ``fit_transform``."""
        self.nstemd.fit_transform(self.signal)
        self.assertIsNotNone(self.nstemd.eig_vec_)
        self.assertIsNotNone(self.nstemd.eig_val_)
        self.assertEqual(self.nstemd.eig_vec_.shape, (3, 3))
        self.assertEqual(self.nstemd.eig_val_.shape, (3,))
        self.assertTrue(np.all(np.diff(self.nstemd.eig_val_) <= 1e-10))

    def test_n_dir_not_overwritten(self) -> None:
        """Constructor ``n_dir`` stays after a decomposition."""
        nstemd = NSTEMD(n_dir=16, max_iter=20)
        nstemd.fit_transform(self.signal)
        self.assertEqual(nstemd.n_dir, 16)

    def test_fit_transform_fix_h(self) -> None:
        """``stop_crit='fix_h'`` also yields a reconstructing decomposition."""
        imfs = NSTEMD(
            stop_crit="fix_h", n_dir=8, max_iter=30, stop_cnt=2
        ).fit_transform(self.signal)
        self.assertEqual(imfs.shape[1:], (self.n_samples, self.n_channels))
        np.testing.assert_allclose(imfs.sum(axis=0).T, self.signal, atol=1e-8)

    def test_four_channel_fallback(self) -> None:
        """n>3 uses doubled uniform Hammersley (MATLAB ``N_dim ~= 3``)."""
        t = np.arange(180, dtype=float) / 180.0
        signal = np.vstack(
            [
                np.cos(2.0 * np.pi * 3.0 * t),
                np.cos(2.0 * np.pi * 7.0 * t),
                np.sin(2.0 * np.pi * 3.0 * t),
                np.sin(2.0 * np.pi * 11.0 * t),
            ]
        )
        imfs = NSTEMD(n_dir=8, max_iter=25).fit_transform(signal)
        self.assertEqual(imfs.shape[1:], (signal.shape[1], 4))
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-8)

    def test_transposes_time_major_input(self) -> None:
        """A MATLAB-style ``(n_samples, n_channels)`` array is accepted."""
        imfs = self.nstemd.fit_transform(self.signal.T)
        self.assertEqual(imfs.shape[1], self.n_samples)
        self.assertEqual(imfs.shape[2], self.n_channels)
        np.testing.assert_allclose(imfs.sum(axis=0), self.signal.T, atol=1e-8)

    def test_rejects_too_few_and_too_many_channels(self) -> None:
        """Channel count must lie in MATLAB ``[3, 16]``."""
        with self.assertRaises(ValueError):
            self.nstemd.fit_transform(np.ones((2, 64)))
        with self.assertRaises(ValueError):
            self.nstemd.fit_transform(np.ones((17, 64)))
        with self.assertRaises(ValueError):
            self.nstemd.fit_transform(np.ones(32))
        with self.assertRaises(ValueError):
            self.nstemd.fit_transform(np.ones((3, 2)))

    def test_init_hammersley_trivariate(self) -> None:
        """Trivariate Hammersley sequence has two coordinates per direction."""
        seq = self.nstemd.init_hammersley(3)
        self.assertEqual(seq.shape, (self.nstemd.n_dir, 2))

    def test_direction_vectors_unit_length(self) -> None:
        """Uniform Hammersley directions remain unit vectors."""
        dirs = self.nstemd.direction_vectors(3)
        self.assertEqual(dirs.shape, (8, 3))
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-10)

    def test_princomp_axis(self) -> None:
        """PCA of energy-along-x data aligns with the first axis."""
        rng = np.random.default_rng(0)
        n = 400
        residue = np.column_stack(
            [
                5.0 * rng.standard_normal(n),
                0.1 * rng.standard_normal(n),
                0.1 * rng.standard_normal(n),
            ]
        )
        coeff, latent = princomp(residue)
        self.assertEqual(coeff.shape, (3, 3))
        self.assertEqual(latent.shape, (3,))
        self.assertGreater(latent[0], latent[1])
        self.assertGreater(abs(coeff[0, 0]), 0.95)
        method_coeff, method_lat = self.nstemd.principal_components(residue)
        np.testing.assert_allclose(np.abs(method_coeff), np.abs(coeff), atol=1e-10)
        np.testing.assert_allclose(method_lat, latent)

    def test_princomp_rejects_bad_shape(self) -> None:
        """``princomp`` requires a 2-D array."""
        with self.assertRaises(ValueError):
            princomp(np.ones(10))

    def test_ellipsoid_directions_unit(self) -> None:
        """Ellipsoid-mapped directions are unit vectors."""
        seq = NSTEMD(n_dir=8).init_hammersley(3)
        coeff, latent = princomp(_power_imbalanced().T)
        dirs = ellipsoid_directions(seq, coeff, latent)
        self.assertEqual(dirs.shape, (8, 3))
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-10)
        method_dirs = self.nstemd.ellipsoid_directions(seq, coeff, latent)
        np.testing.assert_allclose(method_dirs, dirs)

    def test_ellipsoid_rejects_bad_shapes(self) -> None:
        """``ellipsoid_directions`` validates PCA and Hammersley layouts."""
        seq = NSTEMD(n_dir=8).init_hammersley(3)
        with self.assertRaises(ValueError):
            ellipsoid_directions(seq[:, :1], np.eye(3), np.ones(3))
        with self.assertRaises(ValueError):
            ellipsoid_directions(seq, np.eye(2), np.ones(3))
        with self.assertRaises(ValueError):
            ellipsoid_directions(seq, np.eye(3), np.ones(2))

    def test_projection_directions_count(self) -> None:
        """Projection set has ``2 * n_dir`` rows."""
        nstemd = NSTEMD(n_dir=8)
        seq = nstemd.init_hammersley(3)
        coeff, latent = princomp(self.signal.T)
        dirs = nstemd.projection_directions(seq, 3, coeff, latent)
        self.assertEqual(dirs.shape, (16, 3))
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-10)
        uniform = nstemd.direction_vectors(3)
        np.testing.assert_allclose(dirs[8:], uniform)

    def test_projection_directions_higher_dim_duplicates_uniform(self) -> None:
        """For n>3 both halves are the uniform Hammersley set."""
        nstemd = NSTEMD(n_dir=8)
        seq = nstemd.init_hammersley(4)
        coeff, latent = princomp(np.random.default_rng(0).standard_normal((80, 4)))
        dirs = projection_directions(seq, 4, coeff, latent, 8)
        self.assertEqual(dirs.shape, (16, 4))
        np.testing.assert_allclose(dirs[:8], dirs[8:])

    def test_ellipsoid_biases_toward_pc1(self) -> None:
        """Nonuniform samples align more with PC1 than uniform Hammersley."""
        residue = _power_imbalanced().T
        nstemd = NSTEMD(n_dir=16)
        seq = nstemd.init_hammersley(3)
        coeff, latent = princomp(residue)
        pc1 = coeff[:, 0]
        uniform = nstemd.direction_vectors(3)
        adapted = ellipsoid_directions(seq, coeff, latent)
        self.assertGreater(
            np.mean(np.abs(adapted @ pc1)), np.mean(np.abs(uniform @ pc1))
        )

    def test_stop_emd_on_constant(self) -> None:
        """A constant residual has too few extrema on every projection."""
        seq = self.nstemd.init_hammersley(3)
        self.assertTrue(self.nstemd.stop_emd(np.ones((40, 3)), seq, 3))

    def test_stop_emd_on_oscillation(self) -> None:
        """An oscillating residual still has extrema, so sifting continues."""
        seq = self.nstemd.init_hammersley(3)
        self.assertFalse(self.nstemd.stop_emd(self.signal.T, seq, 3))

    def test_stop_returns_envelope(self) -> None:
        """``stop`` returns a boolean flag and an envelope of shape ``(T, C)``."""
        seq = self.nstemd.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        flag, env = self.nstemd.stop(
            self.signal.T, time, seq, self.n_samples, self.n_channels
        )
        self.assertIsInstance(flag, (bool, np.bool_))
        self.assertEqual(env.shape, (self.n_samples, self.n_channels))
        self.assertTrue(np.all(np.isfinite(env)))

    def test_fix_counter(self) -> None:
        """``fix`` increments or resets the consecutive-sifting counter."""
        seq = self.nstemd.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        flag, env, counter = self.nstemd.fix(
            self.signal.T, time, seq, self.n_samples, self.n_channels, counter=0
        )
        self.assertIsInstance(flag, (bool, np.bool_))
        self.assertEqual(env.shape, (self.n_samples, self.n_channels))
        self.assertGreaterEqual(counter, 0)

    def test_local_peaks_used(self) -> None:
        """Sine-wave extrema are recovered by the shared ``local_peaks`` helper."""
        time = np.arange(200, dtype=float)
        x = np.sin(2.0 * np.pi * time / 20.0)
        indmin, indmax = local_peaks(x)
        self.assertGreater(indmin.size, 2)
        self.assertGreater(indmax.size, 2)

    def test_envelope_mean_shape(self) -> None:
        """``envelope_mean`` returns finite envelopes and ``2 * n_dir`` counts."""
        seq = self.nstemd.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        coeff, latent = princomp(self.signal.T)
        env, nem, nzm, amp = envelope_mean(
            self.signal.T,
            time,
            seq,
            self.nstemd.n_dir,
            self.n_samples,
            3,
            coeff,
            latent,
        )
        self.assertEqual(env.shape, (self.n_samples, 3))
        self.assertEqual(nem.shape, (2 * self.nstemd.n_dir,))
        self.assertEqual(nzm.shape, (2 * self.nstemd.n_dir,))
        self.assertEqual(amp.shape, (self.n_samples,))
        self.assertTrue(np.all(np.isfinite(env)))

    def test_power_imbalanced_reconstructs(self) -> None:
        """Unbalanced trivariate data still reconstructs."""
        signal = _power_imbalanced()
        imfs = NSTEMD(n_dir=8, max_iter=25).fit_transform(signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
