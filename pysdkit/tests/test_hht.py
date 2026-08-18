# -*- coding: utf-8 -*-
"""
Unit tests for Hilbert-Huang Transform (HHT).
"""

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysdkit import HHT
from pysdkit._emd import EMD, EEMD, CEEMDAN, REMD
from pysdkit._emd.hht.frequency import get_envelope_frequency, hilbert
from pysdkit.plot import plot_IMFs
from pysdkit.utils import hilbert_spectrum as util_hilbert_spectrum


def _sample_univariate(n_samples: int = 256, sampling_rate: float = 256.0):
    """Two-tone signal similar to the HHT notebook example."""
    time = np.arange(n_samples, dtype=float) / sampling_rate
    signal = np.cos(2.0 * np.pi * 5.0 * time) + 0.5 * np.cos(2.0 * np.pi * 24.0 * time)
    return signal, sampling_rate, time


class HHTTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.HHT` and Hilbert helpers."""

    def setUp(self) -> None:
        self.signal, self.fs, self.time = _sample_univariate()
        self.hht = HHT(algorithm="EMD", max_imfs=4)

    def tearDown(self) -> None:
        plt.close("all")

    def test_str(self) -> None:
        """``str(HHT)`` reports the algorithm name."""
        self.assertEqual(str(self.hht), "Hilbert-Huang Transform (HHT)")

    def test_init_stores_parameters(self) -> None:
        """Constructor stores the requested EMD backend and IMF limit."""
        self.assertEqual(self.hht.algorithm, "EMD")
        self.assertEqual(self.hht.max_imfs, 4)
        self.assertIsInstance(self.hht.emd, EMD)

    def test_import_from_package_root(self) -> None:
        """HHT is exported from the package root."""
        from pysdkit import HHT as RootHHT

        self.assertIs(RootHHT, HHT)

    def test_get_emd_emd(self) -> None:
        """``_get_emd`` returns an EMD instance for algorithm ``EMD``."""
        emd = HHT(algorithm="EMD", max_imfs=3)._get_emd()
        self.assertIsInstance(emd, EMD)
        self.assertEqual(emd.max_imfs, 3)

    def test_get_emd_remd(self) -> None:
        """``_get_emd`` returns a REMD instance for algorithm ``REMD``."""
        emd = HHT(algorithm="REMD", max_imfs=3)._get_emd()
        self.assertIsInstance(emd, REMD)
        self.assertEqual(emd.max_imfs, 3)

    def test_get_emd_eemd(self) -> None:
        """``_get_emd`` returns an EEMD instance for algorithm ``EEMD``."""
        emd = HHT(algorithm="EEMD", max_imfs=2)._get_emd()
        self.assertIsInstance(emd, EEMD)
        self.assertEqual(emd.max_imfs, 2)

    def test_get_emd_ceemdan(self) -> None:
        """``_get_emd`` returns a CEEMDAN instance for algorithm ``CEEMDAN``."""
        emd = HHT(algorithm="CEEMDAN", max_imfs=2)._get_emd()
        self.assertIsInstance(emd, CEEMDAN)
        self.assertEqual(emd.max_imfs, 2)

    def test_get_emd_invalid_algorithm(self) -> None:
        """Unknown algorithm names raise ``ValueError``."""
        with self.assertRaises(ValueError):
            HHT(algorithm="not-an-emd")

    def test_fit_transform_default_returns_imfs(self) -> None:
        """``fit_transform`` returns IMF modes by default."""
        imfs = self.hht.fit_transform(self.signal, fs=self.fs)
        self.assertIsInstance(imfs, np.ndarray)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], self.signal.size)
        self.assertGreaterEqual(imfs.shape[0], 1)
        self.assertLessEqual(imfs.shape[0], 4)

    def test_fit_transform_return_all(self) -> None:
        """``return_all=True`` also returns envelopes and instantaneous frequencies."""
        imfs, envelopes, freqs = self.hht.fit_transform(
            self.signal, fs=self.fs, return_all=True
        )
        self.assertEqual(imfs.shape, envelopes.shape)
        self.assertEqual(imfs.shape, freqs.shape)
        self.assertTrue(np.all(np.isfinite(envelopes)))
        self.assertTrue(np.all(np.isfinite(freqs)))
        np.testing.assert_allclose(
            envelopes, np.abs(hilbert(imfs)), rtol=1e-5, atol=1e-8
        )

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        imfs_fit = self.hht.fit_transform(self.signal, fs=self.fs, return_all=False)
        imfs_call = self.hht(self.signal, fs=self.fs, return_all=False)
        np.testing.assert_allclose(imfs_call, imfs_fit)

    def test_call_return_all(self) -> None:
        """``__call__`` forwards ``return_all`` to ``fit_transform``."""
        result = self.hht(self.signal, fs=self.fs, return_all=True)
        self.assertEqual(len(result), 3)

    def test_save_decompsition_stores_results(self) -> None:
        """``save_decompsition`` records the last decomposition on the instance."""
        imfs, envelopes, freqs = self.hht.fit_transform(
            self.signal, fs=self.fs, return_all=True
        )
        self.assertIs(self.hht.signal, self.signal)
        self.assertEqual(self.hht.fs, self.fs)
        np.testing.assert_array_equal(self.hht.imfs, imfs)
        np.testing.assert_array_equal(self.hht.imfs_env, envelopes)
        np.testing.assert_array_equal(self.hht.imfs_freq, freqs)

        extra = np.arange(self.signal.size, dtype=float)
        extra_imfs = np.vstack([extra, extra])
        extra_env = np.ones_like(extra_imfs)
        extra_freq = np.full_like(extra_imfs, 7.0)
        self.hht.save_decompsition(
            signal=extra,
            fs=128.0,
            imfs=extra_imfs,
            imfs_env=extra_env,
            imfs_freq=extra_freq,
        )
        np.testing.assert_array_equal(self.hht.signal, extra)
        self.assertEqual(self.hht.fs, 128.0)
        np.testing.assert_array_equal(self.hht.imfs, extra_imfs)
        np.testing.assert_array_equal(self.hht.imfs_env, extra_env)
        np.testing.assert_array_equal(self.hht.imfs_freq, extra_freq)

    def test_fit_transform_with_remd(self) -> None:
        """HHT can use REMD as the sifting backend."""
        hht = HHT(algorithm="REMD", max_imfs=3)
        imfs = hht.fit_transform(self.signal, fs=self.fs)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], self.signal.size)
        self.assertIsInstance(hht.emd, REMD)

    def test_plot_imfs_without_signal_raises(self) -> None:
        """``plot_IMFs`` requires a signal when none has been stored."""
        with self.assertRaises(ValueError):
            self.hht.plot_IMFs()

    def test_plot_imfs_after_fit(self) -> None:
        """``plot_IMFs`` uses stored results after ``fit_transform``."""
        self.hht.fit_transform(self.signal, fs=self.fs)
        fig = self.hht.plot_IMFs()
        self.assertIsNotNone(fig)

    def test_plot_imfs_with_explicit_arrays(self) -> None:
        """``plot_IMFs`` accepts an explicit signal and IMF array."""
        imfs = self.hht.fit_transform(self.signal, fs=self.fs)
        fig = self.hht.plot_IMFs(signal=self.signal, imfs=imfs)
        self.assertIsNotNone(fig)
        expected = plot_IMFs(signal=self.signal, IMFs=imfs, return_figure=True)
        self.assertEqual(type(fig), type(expected))

    def test_hilbert_spectrum_shape(self) -> None:
        """``hilbert_spectrum`` returns a 2-D spectrogram and matching axes."""
        _, envelopes, freqs = self.hht.fit_transform(
            self.signal, fs=self.fs, return_all=True
        )
        spec, time_axis, freq_axis = self.hht.hilbert_spectrum(
            imfs_env=envelopes,
            imfs_freq=freqs,
            fs=int(self.fs),
            freq_lim=(0.0, 80.0),
            freq_res=0.5,
        )
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[0], time_axis.size)
        self.assertEqual(spec.shape[1], freq_axis.size)
        self.assertGreater(spec.size, 0)
        self.assertTrue(np.all(np.isfinite(spec)))

        spec_util, t_util, f_util = util_hilbert_spectrum(
            envelopes, freqs, fs=int(self.fs), freq_lim=(0.0, 80.0), freq_res=0.5
        )
        np.testing.assert_allclose(spec, spec_util)
        np.testing.assert_allclose(time_axis, t_util)
        np.testing.assert_allclose(freq_axis, f_util)

    def test_hilbert_spectrum_uses_stored_results(self) -> None:
        """After ``fit_transform``, ``hilbert_spectrum`` can reuse stored envelopes."""
        self.hht.fit_transform(self.signal, fs=self.fs, return_all=True)
        spec, time_axis, freq_axis = self.hht.hilbert_spectrum(
            freq_lim=(0.0, 80.0), freq_res=1.0, time_scale=2
        )
        self.assertEqual(spec.shape[0], time_axis.size)
        self.assertEqual(spec.shape[1], freq_axis.size)

    def test_plot_spectrum_default(self) -> None:
        """``plot_spectrum`` builds a Hilbert spectrogram figure."""
        self.hht.fit_transform(self.signal, fs=self.fs, return_all=True)
        fig = self.hht.plot_spectrum(freq_lim=(0.0, 80.0), freq_res=1.0)
        self.assertIsNotNone(fig)

    def test_plot_spectrum_with_explicit_arrays(self) -> None:
        """``plot_spectrum`` accepts envelopes, frequencies and ``fs`` explicitly."""
        _, envelopes, freqs = self.hht.fit_transform(
            self.signal, fs=self.fs, return_all=True
        )
        fig = self.hht.plot_spectrum(
            imfs_env=envelopes,
            imfs_freq=freqs,
            fs=self.fs,
            freq_lim=(0.0, 80.0),
            freq_res=1.0,
            time_scale=1,
        )
        self.assertIsNotNone(fig)


class HilbertFrequencyTestCase(unittest.TestCase):
    """Tests for Hilbert helpers used by HHT."""

    def test_hilbert_even_length(self) -> None:
        """Even-length inputs produce a complex analytic signal of the same length."""
        signal = np.cos(2.0 * np.pi * np.linspace(0.0, 4.0, 64, endpoint=False))
        analytic = hilbert(signal)
        self.assertEqual(analytic.shape, signal.shape)
        self.assertTrue(np.iscomplexobj(analytic))
        np.testing.assert_allclose(analytic.real, signal, rtol=1e-6, atol=1e-6)
        self.assertGreater(np.mean(np.abs(analytic.imag)), 0.1)

    def test_hilbert_odd_length(self) -> None:
        """Odd-length inputs are also transformed along the last axis."""
        signal = np.cos(2.0 * np.pi * np.linspace(0.0, 4.0, 65, endpoint=False))
        analytic = hilbert(signal)
        self.assertEqual(analytic.shape, signal.shape)
        np.testing.assert_allclose(analytic.real, signal, rtol=1e-6, atol=1e-6)

    def test_hilbert_2d_last_axis(self) -> None:
        """2-D arrays are transformed independently along the last axis."""
        time = np.linspace(0.0, 1.0, 128, endpoint=False)
        modes = np.vstack(
            [np.cos(2.0 * np.pi * 3.0 * time), np.sin(2.0 * np.pi * 7.0 * time)]
        )
        analytic = hilbert(modes)
        self.assertEqual(analytic.shape, modes.shape)
        np.testing.assert_allclose(analytic.real, modes, rtol=1e-6, atol=1e-6)

    def test_get_envelope_frequency_1d(self) -> None:
        """1-D signals return matching envelope and instantaneous-frequency arrays."""
        fs = 256.0
        time = np.arange(256, dtype=float) / fs
        signal = np.cos(2.0 * np.pi * 10.0 * time)
        envelope, freq = get_envelope_frequency(signal, fs=fs)
        self.assertEqual(envelope.shape, signal.shape)
        self.assertEqual(freq.shape, signal.shape)
        self.assertGreater(np.mean(envelope), 0.5)
        self.assertAlmostEqual(float(np.median(freq[20:-20])), 10.0, delta=1.5)

    def test_get_envelope_frequency_return_analytic(self) -> None:
        """``ret_analytic=True`` also returns the complex analytic signal."""
        fs = 128.0
        time = np.arange(128, dtype=float) / fs
        signal = np.cos(2.0 * np.pi * 8.0 * time)
        envelope, freq, analytic = get_envelope_frequency(
            signal, fs=fs, ret_analytic=True
        )
        self.assertEqual(analytic.shape, signal.shape)
        np.testing.assert_allclose(envelope, np.abs(analytic))
        self.assertTrue(np.iscomplexobj(analytic))
        self.assertEqual(freq.shape, signal.shape)

    def test_get_envelope_frequency_2d(self) -> None:
        """IMF stacks keep one envelope/frequency row per mode."""
        fs = 256.0
        time = np.arange(256, dtype=float) / fs
        imfs = np.vstack(
            [np.cos(2.0 * np.pi * 5.0 * time), 0.4 * np.cos(2.0 * np.pi * 20.0 * time)]
        )
        envelope, freq = get_envelope_frequency(imfs, fs=fs)
        self.assertEqual(envelope.shape, imfs.shape)
        self.assertEqual(freq.shape, imfs.shape)
        self.assertGreater(np.mean(envelope[0]), np.mean(envelope[1]))


if __name__ == "__main__":
    unittest.main()
