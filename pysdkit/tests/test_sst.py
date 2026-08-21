# -*- coding: utf-8 -*-
"""
Unit tests for the Synchrosqueezing Transform (SST).
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import SST, sst
from pysdkit._tfa.sst import SST as ModuleSST
from pysdkit._tfa.sst import (
    adaptive_frequency_edges,
    as_channels,
    bump_transfer,
    cwavelet_transform,
    frequency_axis_linear,
    invert_sst_band,
    joint_instantaneous_parameters,
    morlet_transfer,
    multivariate_bandwidth,
    n_scale_components,
    paint_joint_tfr,
    parse_wavelet,
    rcm_thresholds,
    reflect_pad,
    selective_cwt_mask,
    sst_wavelet_linear,
    statistical_mode_int,
    universal_threshold,
)
from pysdkit.data import load_sst_doppler, load_sst_float


def _tone(n_samples: int = 256, freq: float = 0.08, phase: float = 0.0) -> np.ndarray:
    time = np.arange(n_samples, dtype=float)
    return np.cos(2.0 * np.pi * freq * time + phase)


class SSTHelperTest(unittest.TestCase):
    """Tests for module-level SST helpers."""

    def test_parse_wavelet_aliases(self) -> None:
        self.assertEqual(parse_wavelet("bump"), 1)
        self.assertEqual(parse_wavelet(0), 0)
        self.assertEqual(parse_wavelet("MORLET"), 0)

    def test_parse_wavelet_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_wavelet("haar")

    def test_as_channels_1d_and_tall(self) -> None:
        row = as_channels(np.arange(10.0))
        self.assertEqual(row.shape, (1, 10))
        tall = as_channels(np.ones((80, 2)))
        self.assertEqual(tall.shape, (2, 80))

    def test_as_channels_rejects_short_and_3d(self) -> None:
        with self.assertRaises(ValueError):
            as_channels(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            as_channels(np.ones((2, 3, 4)))

    def test_frequency_axis_linear(self) -> None:
        freq = frequency_axis_linear(100)
        self.assertEqual(freq.size, 51)
        self.assertAlmostEqual(freq[0], 0.0)
        self.assertAlmostEqual(freq[-1], 0.5)

    def test_morlet_and_bump_peak(self) -> None:
        omega = np.linspace(0.0, 12.0, 400)
        morlet = morlet_transfer(omega)
        bump = bump_transfer(omega)
        self.assertGreater(morlet[np.argmin(np.abs(omega - 2.0 * np.pi))], 0.0)
        self.assertGreater(bump[np.argmin(np.abs(omega - 5.0))], 0.0)
        self.assertAlmostEqual(bump[0], 0.0)

    def test_reflect_pad(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        padded = reflect_pad(x)
        np.testing.assert_allclose(padded, [4, 3, 2, 1, 1, 2, 3, 4, 1, 2, 3, 4])

    def test_cwavelet_transform_shape(self) -> None:
        x = _tone()
        for wavelet in (0, 1):
            wt, inst, scales, dwt = cwavelet_transform(x, n_voices=8, wavelet=wavelet)
            self.assertEqual(wt.shape[1], x.size)
            self.assertEqual(inst.shape, wt.shape)
            self.assertEqual(dwt.shape, wt.shape)
            self.assertEqual(scales.size, wt.shape[0])
            self.assertTrue(np.all(np.isfinite(scales)))

    def test_cwavelet_transform_rejects_short(self) -> None:
        with self.assertRaises(ValueError):
            cwavelet_transform(np.array([1.0, 2.0]))

    def test_sst_wavelet_linear_concentrates_sine(self) -> None:
        freq0 = 0.08
        x = _tone(n_samples=256, freq=freq0)
        wt, inst, scales, _ = cwavelet_transform(x, n_voices=8, wavelet=1)
        squeezed, freq, tw = sst_wavelet_linear(wt, inst, scales, x)
        self.assertEqual(squeezed.shape[1], x.size)
        self.assertEqual(freq.shape[0], squeezed.shape[0])
        energy = np.mean(np.abs(squeezed) ** 2, axis=1)
        peak = float(freq[int(np.argmax(energy))])
        self.assertLess(abs(peak - freq0), 0.03)
        self.assertEqual(tw.shape, squeezed.shape)

    def test_sst_wavelet_linear_shape_mismatch(self) -> None:
        x = _tone(64)
        wt, inst, scales, _ = cwavelet_transform(x, n_voices=4, wavelet=1)
        with self.assertRaises(ValueError):
            sst_wavelet_linear(wt, inst, scales[:2], x)

    def test_multivariate_bandwidth_tone(self) -> None:
        t = np.arange(200, dtype=float)
        x = np.column_stack(
            [np.cos(2.0 * np.pi * 0.1 * t), np.cos(2.0 * np.pi * 0.1 * t)]
        )
        band, power = multivariate_bandwidth(x)
        self.assertGreater(power, 0.0)
        self.assertGreaterEqual(band, 0.0)
        self.assertTrue(np.isfinite(band))

    def test_multivariate_bandwidth_zero(self) -> None:
        band, power = multivariate_bandwidth(np.zeros((10, 2)))
        self.assertEqual(power, 0.0)
        self.assertEqual(band, 0.0)

    def test_invert_sst_band_empty(self) -> None:
        z = np.ones((5, 8), dtype=np.complex128)
        np.testing.assert_allclose(invert_sst_band(z, 3, 3), np.zeros(8))

    def test_n_scale_components(self) -> None:
        mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
        self.assertEqual(n_scale_components(mask), 3)
        self.assertEqual(n_scale_components(np.zeros(4, dtype=bool)), 0)

    def test_statistical_mode_int(self) -> None:
        self.assertEqual(statistical_mode_int(np.array([1, 2, 2, 3])), 2)
        self.assertEqual(statistical_mode_int(np.array([])), 0)

    def test_rcm_thresholds_and_selective_mask(self) -> None:
        x = _tone(128)
        wt, _, _, _ = cwavelet_transform(x, n_voices=8, wavelet=1)
        gammas = rcm_thresholds(np.abs(wt), n_gamma=12)
        self.assertGreaterEqual(gammas.size, 1)
        masked, gamma, n_modes = selective_cwt_mask(wt, n_window=16, n_gamma=12)
        self.assertEqual(masked.shape, wt.shape)
        self.assertGreater(gamma, 0.0)
        self.assertGreaterEqual(n_modes, 1)
        self.assertGreater(np.count_nonzero(masked), 0)
        self.assertLess(np.count_nonzero(masked), masked.size)

    def test_universal_threshold(self) -> None:
        thr = universal_threshold(100, sigma=1.0, gain=0.2)
        self.assertGreater(thr, 0.0)
        self.assertAlmostEqual(thr, 0.2 * np.sqrt(2.0 * np.log(100)))

    def test_paint_joint_tfr(self) -> None:
        joint_if = np.full((1, 40), 0.1)
        joint_amp = np.ones((1, 40))
        tfr, freq = paint_joint_tfr(joint_if, joint_amp, 40)
        self.assertEqual(tfr.shape[1], 40)
        self.assertGreater(np.max(tfr), 0.0)
        self.assertEqual(freq[0], 0.0)


class SSTTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.SST`."""

    def setUp(self) -> None:
        self.x = _tone(n_samples=256, freq=0.07) + 0.5 * _tone(
            n_samples=256, freq=0.18, phase=0.3
        )
        self.sst = SST(n_voices=8, n_levels=3, wavelet="bump")

    def test_str(self) -> None:
        self.assertEqual(str(self.sst), "Synchrosqueezing Transform (SST)")

    def test_import_from_package_root(self) -> None:
        self.assertIs(SST, ModuleSST)

    def test_init_stores_parameters(self) -> None:
        obj = SST(
            n_voices=16, wavelet="morlet", n_levels=4, n_window=8, denoise_gain=0.15
        )
        self.assertEqual(obj.n_voices, 16)
        self.assertEqual(obj.wavelet, 0)
        self.assertEqual(obj.n_levels, 4)
        self.assertEqual(obj.n_window, 8)
        self.assertEqual(obj.denoise_gain, 0.15)

    def test_invalid_constructor(self) -> None:
        with self.assertRaises(ValueError):
            SST(n_voices=0)
        with self.assertRaises(ValueError):
            SST(n_levels=0)
        with self.assertRaises(ValueError):
            SST(denoise_gain=0.0)

    def test_transform_univariate_shape(self) -> None:
        tfr, freq = self.sst.transform(self.x)
        self.assertEqual(tfr.shape[1], self.x.size)
        self.assertEqual(freq.size, tfr.shape[0])
        self.assertTrue(np.all(np.isfinite(np.abs(tfr))))
        self.assertIsNotNone(self.sst.cwt_)
        self.assertIsNotNone(self.sst.sst_)

    def test_fit_transform_univariate_reconstruction(self) -> None:
        imfs = self.sst.fit_transform(self.x)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], self.x.size)
        self.assertGreaterEqual(imfs.shape[0], 2)
        np.testing.assert_allclose(imfs.sum(axis=0), self.x, atol=1e-8)
        self.assertIsNotNone(self.sst.residue)

    def test_call_matches_fit_transform(self) -> None:
        a = SST(n_voices=8, n_levels=3)(self.x)
        b = SST(n_voices=8, n_levels=3).fit_transform(self.x)
        np.testing.assert_allclose(a, b)

    def test_functional_sst(self) -> None:
        imfs = sst(self.x, n_voices=8, n_levels=3)
        self.assertEqual(imfs.shape[1], self.x.size)

    def test_fit_transform_return_tfr(self) -> None:
        imfs, tfr, freq = self.sst.fit_transform(self.x, return_tfr=True)
        self.assertEqual(imfs.shape[1], self.x.size)
        self.assertEqual(tfr.shape[1], self.x.size)
        self.assertEqual(freq.size, tfr.shape[0])

    def test_multivariate_transform_and_modes(self) -> None:
        t = np.arange(180, dtype=float)
        sig = np.vstack(
            [
                np.cos(2.0 * np.pi * 0.07 * t),
                np.cos(2.0 * np.pi * 0.072 * t),
            ]
        )
        analyser = SST(n_voices=8, n_levels=3, wavelet="bump")
        tfr, freq = analyser.transform(sig)
        self.assertEqual(tfr.shape[1], sig.shape[1])
        self.assertTrue(np.all(np.isfinite(tfr)))
        imfs = SST(n_voices=8, n_levels=3).fit_transform(sig)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1], sig.shape[1])
        self.assertEqual(imfs.shape[2], 2)
        np.testing.assert_allclose(imfs.sum(axis=0).T, sig, atol=1e-8)
        if_traj = analyser.instantaneous_frequency()
        self.assertGreaterEqual(if_traj.ndim, 2)

    def test_denoise_length_and_finite(self) -> None:
        rng = np.random.default_rng(0)
        clean = _tone(200, freq=0.08)
        noisy = clean + 0.4 * rng.standard_normal(clean.size)
        out = SST(n_voices=8, n_levels=3, denoise_gain=0.2).denoise(noisy)
        self.assertEqual(out.shape, noisy.shape)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_denoise_multivariate_shape(self) -> None:
        t = np.arange(160, dtype=float)
        sig = np.vstack(
            [np.cos(2.0 * np.pi * 0.09 * t), np.sin(2.0 * np.pi * 0.09 * t)]
        )
        out = SST(n_voices=8, n_levels=3).denoise(sig)
        self.assertEqual(out.shape, sig.shape)

    def test_selective_sparser_than_full(self) -> None:
        rng = np.random.default_rng(1)
        x = _tone(192, freq=0.1) + 0.8 * rng.standard_normal(192)
        full = SST(n_voices=8, n_levels=3)
        tfr_full, _ = full.transform(x)
        sel = SST(n_voices=8, n_levels=3, n_window=16)
        tfr_sel, _ = sel.selective_transform(x)
        energy_full = float(np.sum(np.abs(tfr_full) ** 2))
        energy_sel = float(np.sum(np.abs(tfr_sel) ** 2))
        self.assertGreater(energy_full, 0.0)
        self.assertGreater(energy_sel, 0.0)
        self.assertLessEqual(np.count_nonzero(tfr_sel), np.count_nonzero(tfr_full) + 1)

    def test_selective_multivariate(self) -> None:
        t = np.arange(128, dtype=float)
        sig = np.vstack([np.cos(2.0 * np.pi * 0.1 * t), np.cos(2.0 * np.pi * 0.1 * t)])
        tfr, freq = SST(n_voices=8, n_window=16).selective_transform(sig)
        self.assertEqual(tfr.shape[1], sig.shape[1])
        self.assertEqual(freq.size, tfr.shape[0])

    def test_instantaneous_frequency_before_fit(self) -> None:
        with self.assertRaises(ValueError):
            SST().instantaneous_frequency()

    def test_adaptive_edges_and_joint_params(self) -> None:
        x = _tone(160, freq=0.1)
        analyser = SST(n_voices=8, n_levels=3)
        analyser.transform(x)
        edges = adaptive_frequency_edges(
            analyser.sst_, analyser.sst_if_, analyser.freq_, n_levels=3
        )
        self.assertGreaterEqual(edges.size, 2)
        self.assertAlmostEqual(edges[-1], 0.0)
        joint_if, joint_amp, x_scale = joint_instantaneous_parameters(
            analyser.sst_, analyser.sst_if_, edges, analyser.freq_
        )
        self.assertEqual(joint_if.shape[1], x.size)
        self.assertEqual(x_scale.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(joint_amp)))

    def test_morlet_wavelet_runs(self) -> None:
        imfs = SST(n_voices=8, n_levels=3, wavelet="morlet").fit_transform(self.x)
        np.testing.assert_allclose(imfs.sum(axis=0), self.x, atol=1e-8)

    def test_packaged_float_snippet(self) -> None:
        record = load_sst_float()
        sig = record["signal"]
        self.assertEqual(sig.shape[0], 2)
        self.assertEqual(sig.shape[1], 1116)
        snippet = sig[:, :160]
        tfr, _ = SST(n_voices=8, n_levels=3).transform(snippet)
        self.assertEqual(tfr.shape[1], 160)
        self.assertTrue(np.all(np.isfinite(tfr)))

    def test_packaged_doppler_loader(self) -> None:
        record = load_sst_doppler()
        self.assertEqual(record["signal"].shape, (2, 2127))


if __name__ == "__main__":
    unittest.main()
