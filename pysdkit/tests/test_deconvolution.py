# -*- coding: utf-8 -*-
"""
Unit tests for IMCKD, ACYCBD, SMHD and shared period helpers.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from pysdkit.data import load_acycbd_sig2, load_imckd_sig1, load_smhd_sig3
from pysdkit.utils.deconvolution import (
    acycbd,
    analytic_envelope,
    annotate_harmonics,
    as_real_1d,
    correlated_kurtosis,
    corr_matrix,
    delay_tensor,
    demean,
    ehps,
    envelope_spectrum,
    estimate_period,
    first_zero_crossing,
    harmonic_label,
    harmonic_peaks,
    imckd,
    marked_envelope_spectrum,
    matlab_kurtosis,
    matlab_round,
    peak_frequency,
    periodic,
    smhd,
    sparse_map,
    xcorr_coeff,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _pulse_train(
    n_samples: int, period: int, fs: float = 1000.0, freq: float = 120.0
) -> np.ndarray:
    """Sparse periodic impacts through a decaying resonant FIR."""
    source = np.zeros(n_samples, dtype=float)
    source[::period] = 1.0
    time = np.arange(80, dtype=float) / fs
    kernel = np.exp(-time / 0.008) * np.sin(2.0 * np.pi * freq * time)
    return np.convolve(source, kernel, mode="same")


class CommonHelperTest(unittest.TestCase):
    """Envelope, ACF, TT, EHPS and MATLAB kurtosis."""

    def test_as_real_1d_column(self) -> None:
        samples = as_real_1d(np.arange(5.0).reshape(-1, 1))
        np.testing.assert_array_equal(samples, np.arange(5.0))

    def test_as_real_1d_rejects_2d(self) -> None:
        with self.assertRaises(ValueError):
            as_real_1d(np.ones((2, 3)))

    def test_as_real_1d_rejects_complex(self) -> None:
        with self.assertRaises(ValueError):
            as_real_1d(np.ones(4) + 1j)

    def test_demean_zero_mean(self) -> None:
        samples = demean(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.mean(samples)), 0.0, places=12)

    def test_analytic_envelope_mean_zero(self) -> None:
        envelope = analytic_envelope(_pulse_train(512, 40))
        self.assertAlmostEqual(float(np.mean(envelope)), 0.0, delta=1e-10)

    def test_matlab_round_ties_away_from_zero(self) -> None:
        self.assertEqual(matlab_round(2.5), 3)
        self.assertEqual(matlab_round(-2.5), -3)
        self.assertEqual(matlab_round(50.0), 50)

    def test_matlab_kurtosis_gaussian_near_three(self) -> None:
        samples = _rng(1).normal(size=40_000)
        self.assertAlmostEqual(matlab_kurtosis(samples), 3.0, delta=0.08)

    def test_xcorr_coeff_lag_zero_is_one(self) -> None:
        samples = _rng(2).normal(size=64)
        acf = xcorr_coeff(samples, max_lag=10)
        self.assertEqual(acf.size, 11)
        self.assertAlmostEqual(float(acf[0]), 1.0, places=10)

    def test_xcorr_coeff_zero_signal(self) -> None:
        acf = xcorr_coeff(np.zeros(16), max_lag=4)
        self.assertAlmostEqual(float(acf[0]), 1.0)
        np.testing.assert_array_equal(acf[1:], np.zeros(4))

    def test_xcorr_coeff_rejects_negative_lag(self) -> None:
        with self.assertRaises(ValueError):
            xcorr_coeff(np.ones(8), max_lag=-1)

    def test_first_zero_crossing_sign_change(self) -> None:
        acf = np.array([1.0, 0.4, 0.1, -0.2, -0.1])
        self.assertEqual(first_zero_crossing(acf), 3)

    def test_first_zero_crossing_exact_zero(self) -> None:
        acf = np.array([1.0, 0.0, -0.2])
        self.assertEqual(first_zero_crossing(acf), 1)

    def test_first_zero_crossing_never_raises(self) -> None:
        with self.assertRaises(ValueError):
            first_zero_crossing(np.array([1.0, 0.9, 0.8, 0.7]))

    def test_first_zero_crossing_too_short(self) -> None:
        with self.assertRaises(ValueError):
            first_zero_crossing(np.array([1.0]))

    def test_estimate_period_near_known_lag(self) -> None:
        period = 40
        samples = _pulse_train(800, period)
        lag, hnr = estimate_period(analytic_envelope(samples), fs=200)
        self.assertGreater(hnr, 0.0)
        self.assertLess(abs((lag - 2) - period), 6)

    def test_envelope_spectrum_scale_and_peak(self) -> None:
        period = 50
        fs = 1000.0
        samples = _pulse_train(1000, period, fs=fs)
        freq, mag = envelope_spectrum(samples, fs, scale="length")
        self.assertEqual(freq.size, samples.size)
        peak = peak_frequency(freq, mag, f_max=80.0)
        self.assertAlmostEqual(peak, fs / period, delta=8.0)
        _freq_fs, mag_fs = envelope_spectrum(samples, fs, scale="fs")
        self.assertGreater(float(np.max(mag_fs)), 0.0)

    def test_envelope_spectrum_bad_scale(self) -> None:
        with self.assertRaises(ValueError):
            envelope_spectrum(np.ones(16), fs=1.0, scale="bad")

    def test_peak_frequency_empty_band(self) -> None:
        with self.assertRaises(ValueError):
            peak_frequency(np.array([0.0, 1.0]), np.array([1.0, 2.0]), f_max=0.5)

    def test_ehps_finds_envelope_fundamental(self) -> None:
        fs = 1000.0
        n_samples = 1000
        time = np.arange(n_samples, dtype=float) / fs
        carrier = np.sin(2.0 * np.pi * 180.0 * time)
        envelope = np.abs(np.sin(np.pi * 40.0 * time)) ** 4
        estimated = ehps(carrier * envelope, fs, n_harmonics=5, flim=80.0)
        self.assertAlmostEqual(estimated, 40.0, delta=2.0)

    def test_ehps_pulse_train(self) -> None:
        fs = 1000.0
        estimated = ehps(_pulse_train(1000, 25, fs=fs), fs, n_harmonics=5, flim=80.0)
        self.assertAlmostEqual(estimated, 40.0, delta=2.0)

    def test_ehps_rejects_short_record(self) -> None:
        with self.assertRaises(ValueError):
            ehps(np.array([1.0]), fs=10.0)


class IMCKDTest(unittest.TestCase):
    """Delay tensor, correlated kurtosis, and IMCKD solver."""

    def test_delay_tensor_known_shifts(self) -> None:
        samples = np.arange(1.0, 6.0)
        tensor = delay_tensor(samples, filter_size=3, period=2, shift_order=1)
        self.assertEqual(tensor.shape, (3, 5, 2))
        np.testing.assert_array_equal(tensor[0, :, 0], samples)
        np.testing.assert_array_equal(
            tensor[1, :, 0], np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        )
        np.testing.assert_array_equal(
            tensor[0, :, 1], np.array([0.0, 0.0, 1.0, 2.0, 3.0])
        )

    def test_delay_tensor_rejects_bad_size(self) -> None:
        with self.assertRaises(ValueError):
            delay_tensor(np.ones(8), filter_size=0, period=1, shift_order=1)

    def test_correlated_kurtosis_finite(self) -> None:
        samples = _pulse_train(256, 32)
        value = correlated_kurtosis(samples, period=32, shift_order=1)
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_imckd_synthetic_unit_filter(self) -> None:
        samples = _pulse_train(400, 32)
        filtered, fir, info = imckd(
            samples,
            fs=80,
            filter_size=8,
            term_iter=3,
            shift_order=1,
        )
        self.assertEqual(filtered.size, samples.size)
        self.assertEqual(fir.size, 8)
        self.assertAlmostEqual(float(np.linalg.norm(fir)), 1.0, places=6)
        self.assertEqual(np.asarray(info["ck_iter"]).size, 3)
        self.assertEqual(np.asarray(info["period_hist"]).size, 4)
        self.assertEqual(np.asarray(info["kurtosis_hist"]).size, 4)
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_imckd_rejects_tiny_filter(self) -> None:
        with self.assertRaises(ValueError):
            imckd(np.ones(32), fs=16, filter_size=1)

    def test_imckd_demo_envelope_peak(self) -> None:
        record = load_imckd_sig1()
        samples = record["signal"] - np.mean(record["signal"])
        filtered, fir, info = imckd(
            samples,
            fs=record["fs"],
            filter_size=50,
            term_iter=30,
            shift_order=1,
        )
        self.assertEqual(filtered.size, samples.size)
        self.assertAlmostEqual(float(np.linalg.norm(fir)), 1.0, places=5)
        freq, mag = envelope_spectrum(filtered, record["fs"], scale="fs")
        peak = peak_frequency(freq, mag, f_max=200.0)
        self.assertAlmostEqual(peak, float(record["fault_hz"]), delta=8.0)
        self.assertGreater(float(info["kurtosis"]), 0.0)


class ACYCBDTest(unittest.TestCase):
    """Weighted correlation, periodic projection, and ACYCBD solver."""

    def test_corr_matrix_hermitian(self) -> None:
        samples = _rng(3).normal(size=64)
        gram = corr_matrix(samples, None, n_taps=6)
        self.assertEqual(gram.shape, (6, 6))
        np.testing.assert_allclose(gram, gram.T, atol=1e-12)

    def test_corr_matrix_weighted_length(self) -> None:
        samples = _rng(4).normal(size=32)
        n_taps = 5
        weights = np.linspace(0.2, 1.0, 32 - n_taps + 1)
        gram = corr_matrix(samples, weights, n_taps)
        self.assertEqual(gram.shape, (n_taps, n_taps))
        self.assertTrue(np.all(np.isfinite(gram)))

    def test_corr_matrix_rejects_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            corr_matrix(np.ones(4), None, n_taps=8)

    def test_corr_matrix_rejects_bad_weights(self) -> None:
        with self.assertRaises(ValueError):
            corr_matrix(np.ones(16), np.ones(3), n_taps=4)

    def test_periodic_length_and_gate(self) -> None:
        fs = 200.0
        time = np.arange(400, dtype=float) / fs
        samples = 0.5 + np.cos(2.0 * np.pi * 10.0 * time)
        gated = periodic(samples, np.array([10.0, 20.0]), fs)
        self.assertEqual(gated.size, samples.size)
        self.assertTrue(np.all(np.isfinite(gated)))
        self.assertGreaterEqual(float(np.min(gated)), 0.0)

    def test_acycbd_shortens_by_filter_minus_one(self) -> None:
        samples = _pulse_train(512, 40)
        fir, recovered, info = acycbd(
            samples, fs=200.0, filter_size=8, max_iter=3, n_harmonics=4
        )
        self.assertEqual(fir.size, 8)
        self.assertEqual(recovered.size, samples.size - 7)
        self.assertEqual(int(info["count"]), np.asarray(info["f_est"]).size)
        self.assertTrue(np.all(np.isfinite(recovered)))
        self.assertEqual(np.asarray(info["err"])[0], np.inf)

    def test_acycbd_rejects_tiny_filter(self) -> None:
        with self.assertRaises(ValueError):
            acycbd(np.ones(32), fs=16.0, filter_size=1)

    def test_acycbd_demo_cyclic_frequency(self) -> None:
        record = load_acycbd_sig2()
        samples = record["signal"] - np.mean(record["signal"])
        _fir, recovered, info = acycbd(samples, fs=record["fs"], filter_size=40)
        self.assertEqual(recovered.size, samples.size - 39)
        last_freq = float(np.asarray(info["f_est"])[-1])
        self.assertAlmostEqual(last_freq, float(record["bpfi"]), delta=8.0)


class SMHDTest(unittest.TestCase):
    """Sparsity map and SMHD solver."""

    def test_sparse_map_shrinks_amplitude(self) -> None:
        samples = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        mapped = sparse_map(samples, mu=1.0)
        self.assertEqual(mapped.size, samples.size)
        np.testing.assert_array_less(np.abs(mapped) - 1e-15, np.abs(samples) + 1e-15)
        self.assertAlmostEqual(float(mapped[2]), 0.0)
        self.assertGreater(float(mapped[4]), 0.0)
        self.assertLess(float(mapped[0]), 0.0)

    def test_sparse_map_zero_mu(self) -> None:
        mapped = sparse_map(np.ones(4), mu=0.0)
        np.testing.assert_array_equal(mapped, np.zeros(4))

    def test_smhd_synthetic_unit_filter(self) -> None:
        samples = _pulse_train(400, 32)
        filtered, fir, info = smhd(
            samples,
            fs=80,
            filter_size=8,
            term_iter=3,
            mu=1.5 * float(np.sqrt(np.mean(samples**2))),
        )
        self.assertEqual(filtered.size, samples.size)
        self.assertAlmostEqual(float(np.linalg.norm(fir)), 1.0, places=6)
        self.assertEqual(np.asarray(info["kurt_iter"]).size, 3)
        self.assertEqual(np.asarray(info["hnr"]).size, 3)
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_smhd_rejects_tiny_filter(self) -> None:
        with self.assertRaises(ValueError):
            smhd(np.ones(32), fs=16, filter_size=1)

    def test_smhd_demo_envelope_peak(self) -> None:
        record = load_smhd_sig3()
        samples = record["signal"] - np.mean(record["signal"])
        rms = float(np.sqrt(np.mean(samples**2)))
        filtered, fir, info = smhd(
            samples,
            fs=record["fs"],
            filter_size=100,
            term_iter=30,
            mu=1.5 * rms,
        )
        self.assertEqual(filtered.size, samples.size)
        self.assertAlmostEqual(float(np.linalg.norm(fir)), 1.0, places=5)
        freq, mag = envelope_spectrum(filtered, record["fs"], scale="fs")
        peak = peak_frequency(freq, mag, f_max=200.0)
        self.assertAlmostEqual(peak, float(record["bpfi"]), delta=8.0)
        self.assertGreater(float(info["hnr_max"]), 0.0)


class HarmonicAnnotateTest(unittest.TestCase):
    """Paper-style f_o, 2f_o, … markers on an envelope spectrum."""

    def _line_spectrum(self) -> tuple:
        freq = np.linspace(0.0, 300.0, 3001)
        magnitude = 0.02 * np.ones_like(freq)
        fund = 30.0
        for order in range(1, 7):
            magnitude += (
                0.15 / order * np.exp(-0.5 * ((freq - order * fund) / 0.8) ** 2)
            )
        return freq, magnitude, fund

    def test_harmonic_label_fundamental_and_multiple(self) -> None:
        self.assertEqual(harmonic_label(1), r"$f_o$")
        self.assertEqual(harmonic_label(3), r"$3f_o$")
        self.assertEqual(harmonic_label(2, symbol=r"f_i"), r"$2f_i$")

    def test_harmonic_label_rejects_zero_order(self) -> None:
        with self.assertRaises(ValueError):
            harmonic_label(0)

    def test_harmonic_peaks_on_synthetic_lines(self) -> None:
        freq, magnitude, fund = self._line_spectrum()
        peaks = harmonic_peaks(freq, magnitude, fund, n_harmonics=6, f_max=200.0)
        self.assertEqual(peaks.size, 6)
        np.testing.assert_array_equal(peaks["order"], np.arange(1, 7))
        for row in peaks:
            self.assertAlmostEqual(
                float(row["frequency"]), row["order"] * fund, delta=1.0
            )

    def test_harmonic_peaks_rejects_nonpositive_fundamental(self) -> None:
        freq = np.linspace(0.0, 10.0, 32)
        with self.assertRaises(ValueError):
            harmonic_peaks(freq, np.ones_like(freq), 0.0)

    def test_annotate_harmonics_on_axes(self) -> None:
        freq, magnitude, fund = self._line_spectrum()
        fig, axes = plt.subplots()
        axes.plot(freq, magnitude)
        axes.set_xlim(0, 220)
        returned, peaks, notes = annotate_harmonics(
            freq, magnitude, fund, n_harmonics=6, ax=axes, f_max=200.0
        )
        self.assertIs(returned, axes)
        self.assertEqual(peaks.size, 6)
        self.assertEqual(len(notes), 6)
        self.assertEqual(notes[0].get_text(), r"$f_o$")
        self.assertEqual(notes[1].get_text(), r"$2f_o$")
        plt.close(fig)

    def test_marked_envelope_spectrum_infers_fundamental(self) -> None:
        freq, magnitude, fund = self._line_spectrum()
        fig, axes = plt.subplots()
        returned, peaks = marked_envelope_spectrum(
            freq, magnitude, n_harmonics=4, ax=axes, f_max=140.0
        )
        self.assertIs(returned, axes)
        self.assertGreaterEqual(peaks.size, 1)
        self.assertAlmostEqual(float(peaks["frequency"][0]), fund, delta=1.5)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
