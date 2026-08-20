# -*- coding: utf-8 -*-
"""
Unit tests for Adaptive-Projection Intrinsically Transformed MEMD (APIT-MEMD).
"""

import unittest

import numpy as np

from pysdkit import APITMEMD, MEMD
from pysdkit._emd.apitmemd import APITMEMD as ModuleAPITMEMD
from pysdkit._emd.apitmemd import (
    check_multivariate_signal,
    envelope_mean,
    first_principal_component,
    hammersley_unit_directions,
    local_peaks,
    nonuniform_directions,
    principal_components,
)
from pysdkit.data import (
    load_apitmemd_section_2b,
    load_apitmemd_section_3a,
    load_apitmemd_section_3b,
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


def _power_imbalanced(n_samples: int = 220) -> np.ndarray:
    """Trivariate record whose energy is concentrated on one axis."""
    time = np.arange(n_samples, dtype=float) / float(n_samples)
    strong = np.cos(2.0 * np.pi * 6.0 * time) + 0.3 * np.cos(2.0 * np.pi * 19.0 * time)
    weak = 0.05 * np.sin(2.0 * np.pi * 6.0 * time)
    weaker = 0.02 * np.cos(2.0 * np.pi * 11.0 * time)
    return np.vstack([10.0 * strong, weak, weaker])


class APITMEMDTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.APITMEMD` and module-level helpers."""

    def setUp(self) -> None:
        self.signal = _trivariate_tones()
        self.n_channels, self.n_samples = self.signal.shape
        self.apit = APITMEMD(n_dir=8, max_iter=40, stop_cnt=2, alpha=0.3)

    def test_str(self) -> None:
        """``str(APITMEMD)`` reports the algorithm name."""
        self.assertEqual(
            str(self.apit),
            "Adaptive-Projection Intrinsically Transformed MEMD (APIT-MEMD)",
        )

    def test_init_stores_parameters(self) -> None:
        """Constructor stores stopping, projection and alpha parameters."""
        apit = APITMEMD(
            stop_crit="fix_h",
            max_iter=50,
            n_dir=12,
            stop_vec=[0.05, 0.5, 0.05],
            stop_cnt=3,
            alpha=0.7,
        )
        self.assertEqual(apit.stop_crit, "fix_h")
        self.assertEqual(apit.max_iter, 50)
        self.assertEqual(apit.n_dir, 12)
        self.assertEqual(apit.stop_cnt, 3)
        self.assertEqual(apit.alpha, 0.7)
        np.testing.assert_allclose(apit.stop_vec, [0.05, 0.5, 0.05])
        self.assertEqual(apit.sd, 0.05)
        self.assertEqual(apit.sd2, 0.5)
        self.assertEqual(apit.tol, 0.05)

    def test_default_alpha(self) -> None:
        """MATLAB default ``alpha`` is 0.3."""
        self.assertEqual(APITMEMD().alpha, 0.3)

    def test_import_from_package_root(self) -> None:
        """APITMEMD is exported from the package root and the ``_emd`` module."""
        self.assertIs(APITMEMD, ModuleAPITMEMD)

    def test_invalid_alpha(self) -> None:
        """Negative or non-finite ``alpha`` is rejected."""
        with self.assertRaises(ValueError):
            APITMEMD(alpha=-0.1)
        with self.assertRaises(ValueError):
            APITMEMD(alpha=np.nan)
        with self.assertRaises(ValueError):
            APITMEMD(alpha=np.inf)

    def test_invalid_n_dir(self) -> None:
        """Fewer than 6 direction vectors is rejected."""
        with self.assertRaises(ValueError):
            APITMEMD(n_dir=5)

    def test_invalid_stop_crit(self) -> None:
        """Unknown stopping criteria raise ``ValueError``."""
        with self.assertRaises(ValueError):
            APITMEMD(stop_crit="unknown")

    def test_fit_transform_shape(self) -> None:
        """``fit_transform`` returns ``(K, T, C)`` with finite values."""
        imfs = self.apit.fit_transform(self.signal)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1], self.n_samples)
        self.assertEqual(imfs.shape[2], self.n_channels)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        """Summing IMFs including the residue recovers the input."""
        imfs = self.apit.fit_transform(self.signal)
        reconstructed = imfs.sum(axis=0).T
        np.testing.assert_allclose(reconstructed, self.signal, atol=1e-8)

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        a = self.apit(self.signal)
        b = self.apit.fit_transform(self.signal)
        np.testing.assert_allclose(a, b)

    def test_n_dir_and_alpha_not_overwritten(self) -> None:
        """Constructor values stay after a decomposition."""
        apit = APITMEMD(n_dir=16, max_iter=20, alpha=0.5)
        apit.fit_transform(self.signal)
        self.assertEqual(apit.n_dir, 16)
        self.assertEqual(apit.alpha, 0.5)

    def test_fit_transform_fix_h(self) -> None:
        """``stop_crit='fix_h'`` also yields a reconstructing decomposition."""
        imfs = APITMEMD(
            stop_crit="fix_h", n_dir=8, max_iter=30, stop_cnt=2, alpha=0.3
        ).fit_transform(self.signal)
        self.assertEqual(imfs.shape[1:], (self.n_samples, self.n_channels))
        np.testing.assert_allclose(imfs.sum(axis=0).T, self.signal, atol=1e-8)

    def test_alpha_zero_reconstructs(self) -> None:
        """``alpha=0`` (MEMD-like relocation off) still reconstructs."""
        imfs = APITMEMD(n_dir=8, max_iter=25, alpha=0.0).fit_transform(self.signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, self.signal, atol=1e-8)

    def test_four_channel_input(self) -> None:
        """4-channel data uses the n-sphere Hammersley map plus APIT relocation."""
        t = np.arange(180, dtype=float) / 180.0
        signal = np.vstack(
            [
                np.cos(2.0 * np.pi * 3.0 * t),
                np.cos(2.0 * np.pi * 7.0 * t),
                np.sin(2.0 * np.pi * 3.0 * t),
                np.sin(2.0 * np.pi * 11.0 * t),
            ]
        )
        imfs = APITMEMD(n_dir=8, max_iter=25, alpha=0.3).fit_transform(signal)
        self.assertEqual(imfs.shape[1:], (signal.shape[1], 4))
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-8)

    def test_transposes_time_major_input(self) -> None:
        """A MATLAB-style ``(n_samples, n_channels)`` array is accepted."""
        imfs = self.apit.fit_transform(self.signal.T)
        self.assertEqual(imfs.shape[1], self.n_samples)
        self.assertEqual(imfs.shape[2], self.n_channels)
        np.testing.assert_allclose(imfs.sum(axis=0), self.signal.T, atol=1e-8)

    def test_rejects_too_few_channels(self) -> None:
        """Univariate / bivariate inputs raise ``ValueError``."""
        with self.assertRaises(ValueError):
            self.apit.fit_transform(np.ones((2, 64)))

    def test_rejects_too_many_channels(self) -> None:
        """More than 32 channels is outside the MATLAB toolbox limit."""
        with self.assertRaises(ValueError):
            self.apit.fit_transform(np.ones((33, 64)))

    def test_allows_more_channels_than_memd(self) -> None:
        """APIT-MEMD accepts 17–32 channels; MEMD stops at 16."""
        x17 = np.ones((17, 40))
        with self.assertRaises(ValueError):
            MEMD(n_dir=8).fit_transform(x17)
        checked = check_multivariate_signal(x17)
        self.assertEqual(checked.shape, (17, 40))

    def test_rejects_1d_and_short_series(self) -> None:
        """Non-2-D arrays and very short records are rejected."""
        with self.assertRaises(ValueError):
            self.apit.fit_transform(np.ones(32))
        with self.assertRaises(ValueError):
            self.apit.fit_transform(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            check_multivariate_signal(np.ones(16))

    def test_init_hammersley_trivariate(self) -> None:
        """Trivariate Hammersley sequence has two coordinates per direction."""
        seq = self.apit.init_hammersley(3)
        self.assertEqual(seq.shape, (self.apit.n_dir, 2))
        self.assertTrue(np.all(np.isfinite(seq)))

    def test_init_hammersley_higher_dim(self) -> None:
        """For n>3 the sequence has one Hammersley coordinate per dimension."""
        seq = APITMEMD(n_dir=8).init_hammersley(5)
        self.assertEqual(seq.shape, (8, 5))

    def test_direction_vectors_unit_length(self) -> None:
        """Each Hammersley direction is a unit vector."""
        for n_dim in (3, 4, 6):
            dirs = APITMEMD(n_dir=8).direction_vectors(n_dim)
            self.assertEqual(dirs.shape, (8, n_dim))
            np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-10)

    def test_hammersley_unit_directions(self) -> None:
        """Module helper matches :meth:`direction_vectors`."""
        apit = APITMEMD(n_dir=8)
        seq = apit.init_hammersley(3)
        dirs = hammersley_unit_directions(seq, 8, 3)
        np.testing.assert_allclose(dirs, apit.direction_vectors(3))

    def test_first_principal_component_axis(self) -> None:
        """PC1 of energy-along-x data aligns with the first axis."""
        rng = np.random.default_rng(0)
        n = 400
        residue = np.column_stack(
            [
                5.0 * rng.standard_normal(n),
                0.1 * rng.standard_normal(n),
                0.1 * rng.standard_normal(n),
            ]
        )
        pc1 = first_principal_component(residue)
        self.assertEqual(pc1.shape, (3,))
        np.testing.assert_allclose(np.linalg.norm(pc1), 1.0, atol=1e-12)
        self.assertGreater(abs(pc1[0]), 0.95)

        method_pc1 = self.apit.first_principal_component(residue)
        np.testing.assert_allclose(np.abs(method_pc1), np.abs(pc1), atol=1e-10)

    def test_principal_components_sorted(self) -> None:
        """Eigenvalues are returned in descending order."""
        residue = _power_imbalanced().T
        vecs, vals = principal_components(residue)
        self.assertEqual(vecs.shape, (3, 3))
        self.assertEqual(vals.shape, (3,))
        self.assertTrue(np.all(np.diff(vals) <= 1e-12))
        np.testing.assert_allclose(
            np.abs(vecs[:, 0]), np.abs(first_principal_component(residue))
        )

    def test_nonuniform_directions_unit_and_count(self) -> None:
        """Adapted directions stay unit-length and have even cardinality."""
        uniform = APITMEMD(n_dir=8).direction_vectors(3)
        residue = _power_imbalanced().T
        adapted = nonuniform_directions(uniform, residue, alpha=0.5)
        self.assertEqual(adapted.shape, (8, 3))
        np.testing.assert_allclose(np.linalg.norm(adapted, axis=1), 1.0, atol=1e-10)

        odd = APITMEMD(n_dir=7).direction_vectors(3)
        adapted_odd = nonuniform_directions(odd, residue, alpha=0.3)
        self.assertEqual(adapted_odd.shape, (6, 3))

    def test_adaptive_directions_bias_toward_pc1(self) -> None:
        """Relocation increases alignment with the first principal axis."""
        residue = _power_imbalanced().T
        apit = APITMEMD(n_dir=16, alpha=0.8)
        uniform = apit.direction_vectors(3)
        adapted = apit.adaptive_directions(residue, 3)
        pc1 = first_principal_component(residue)
        mean_uniform = np.mean(np.abs(uniform @ pc1))
        mean_adapted = np.mean(np.abs(adapted @ pc1))
        self.assertGreater(mean_adapted, mean_uniform)

    def test_alpha_zero_keeps_selected_subset_on_sphere(self) -> None:
        """``alpha=0`` re-normalises the PC1-nearest Hammersley subset only."""
        uniform = APITMEMD(n_dir=8).direction_vectors(3)
        residue = self.signal.T
        adapted = nonuniform_directions(uniform, residue, alpha=0.0)
        dots = adapted @ uniform.T
        for row in dots:
            self.assertGreater(np.max(np.abs(row)), 1.0 - 1e-8)

    def test_stop_emd_on_constant(self) -> None:
        """A constant residual has too few extrema on every projection."""
        seq = self.apit.init_hammersley(3)
        constant = np.ones((40, 3))
        self.assertTrue(self.apit.stop_emd(constant, seq, 3))

    def test_stop_emd_on_oscillation(self) -> None:
        """An oscillating residual still has extrema, so sifting continues."""
        seq = self.apit.init_hammersley(3)
        self.assertFalse(self.apit.stop_emd(self.signal.T, seq, 3))

    def test_stop_returns_envelope(self) -> None:
        """``stop`` returns a boolean flag and an envelope of shape ``(T, C)``."""
        seq = self.apit.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        flag, env = self.apit.stop(
            self.signal.T, time, seq, self.n_samples, self.n_channels
        )
        self.assertIsInstance(flag, (bool, np.bool_))
        self.assertEqual(env.shape, (self.n_samples, self.n_channels))
        self.assertTrue(np.all(np.isfinite(env)))

    def test_fix_counter(self) -> None:
        """``fix`` increments or resets the consecutive-sifting counter."""
        seq = self.apit.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        flag, env, counter = self.apit.fix(
            self.signal.T, time, seq, self.n_samples, self.n_channels, counter=0
        )
        self.assertIsInstance(flag, (bool, np.bool_))
        self.assertEqual(env.shape, (self.n_samples, self.n_channels))
        self.assertIsInstance(counter, int)
        self.assertGreaterEqual(counter, 0)

    def test_local_peaks_sine(self) -> None:
        """Sine-wave extrema are recovered by APIT ``local_peaks``."""
        time = np.arange(200, dtype=float)
        x = np.sin(2.0 * np.pi * time / 20.0)
        indmin, indmax = local_peaks(x)
        self.assertGreater(indmin.size, 2)
        self.assertGreater(indmax.size, 2)
        self.assertTrue(np.all(x[indmax] > 0.5))
        self.assertTrue(np.all(x[indmin] < -0.5))

    def test_local_peaks_flat_uses_1e_minus_6(self) -> None:
        """Projections below ``1e-6`` are treated as identically zero."""
        indmin, indmax = local_peaks(np.zeros(50))
        self.assertEqual(indmin.size, 0)
        self.assertEqual(indmax.size, 0)
        tiny = np.full(50, 5e-7)
        a, b = local_peaks(tiny)
        self.assertEqual(a.size, 0)
        self.assertEqual(b.size, 0)

    def test_envelope_mean_shape(self) -> None:
        """``envelope_mean`` returns finite envelopes and extrema counts."""
        seq = self.apit.init_hammersley(3)
        time = np.arange(1, self.n_samples + 1, dtype=float)
        env, nem, nzm, amp = envelope_mean(
            self.signal.T,
            time,
            seq,
            self.apit.n_dir,
            self.n_samples,
            3,
            self.apit.alpha,
        )
        self.assertEqual(env.shape, (self.n_samples, 3))
        self.assertEqual(nem.shape, (self.apit.n_dir,))
        self.assertEqual(nzm.shape, (self.apit.n_dir,))
        self.assertEqual(amp.shape, (self.n_samples,))
        self.assertTrue(np.all(np.isfinite(env)))
        self.assertTrue(np.all(nem >= 0))

    def test_nonuniform_rejects_bad_shapes(self) -> None:
        """``nonuniform_directions`` validates array layouts."""
        uniform = APITMEMD(n_dir=8).direction_vectors(3)
        with self.assertRaises(ValueError):
            nonuniform_directions(uniform.ravel(), self.signal.T, 0.3)
        with self.assertRaises(ValueError):
            nonuniform_directions(uniform, self.signal, 0.3)

    def test_packaged_section_2b_loader(self) -> None:
        """Packaged hexavariate P300 example keeps channel-major layout."""
        demo = load_apitmemd_section_2b()
        self.assertEqual(demo["signal"].shape, (6, 360))
        self.assertEqual(demo["t"].shape, (360,))
        self.assertEqual(demo["fs"], 1200.0)
        self.assertTrue(np.all(np.isfinite(demo["signal"])))

    def test_packaged_section_2b_smoke(self) -> None:
        """Section-2b hexavariate P300 reconstructs after APIT-MEMD."""
        signal = load_apitmemd_section_2b()["signal"]
        imfs = APITMEMD(n_dir=8, max_iter=20, alpha=0.3).fit_transform(signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-7)

    def test_packaged_section_3a_loader(self) -> None:
        """Packaged SSVEP snippet keeps channel-major layout and ``fs=1200``."""
        demo = load_apitmemd_section_3a()
        self.assertEqual(demo["signal"].shape, (4, 2048))
        self.assertEqual(demo["t"].shape, (2048,))
        self.assertEqual(demo["fs"], 1200.0)
        self.assertTrue(np.all(np.isfinite(demo["signal"])))

    def test_packaged_section_3a_smoke(self) -> None:
        """A short slice of the packaged supplement record reconstructs."""
        signal = load_apitmemd_section_3a()["signal"][:, :256]
        imfs = APITMEMD(n_dir=8, max_iter=20, alpha=0.3).fit_transform(signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-7)

    def test_packaged_section_3b_loader(self) -> None:
        """Packaged P300 trials keep ``(n_trials, n_channels, n_samples)``."""
        demo = load_apitmemd_section_3b()
        self.assertEqual(demo["signal"].shape, (10, 6, 240))
        self.assertEqual(demo["t"].shape, (240,))
        self.assertEqual(demo["fs"], 1200.0)
        self.assertTrue(np.all(np.isfinite(demo["signal"])))

    def test_packaged_section_3b_smoke(self) -> None:
        """The first single-shot P300 trial reconstructs after APIT-MEMD."""
        signal = load_apitmemd_section_3b()["signal"][0]
        imfs = APITMEMD(n_dir=8, max_iter=20, alpha=0.3).fit_transform(signal)
        np.testing.assert_allclose(imfs.sum(axis=0).T, signal, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
