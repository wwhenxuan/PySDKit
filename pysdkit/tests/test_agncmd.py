# -*- coding: utf-8 -*-
"""
Unit tests for Adaptive Generalized Dispersive Mode Decomposition
(AGDMD / AGNCMD).
"""

import unittest

import numpy as np

from pysdkit import AGNCMD, AGDMD, agncmd, agdmd
from pysdkit._gdmd.agncmd import (
    agdi,
    agdmd_core,
    arccos_phase,
    bandwidth_estimation,
    ddgdi,
    differ_complex,
    dispersion_compensation,
    findev,
    if_dn,
    low_filter,
    make_agncmd_demo_signal,
    spectrum_to_time_agdmd,
    stft_agncmd,
    tvlp,
)
from pysdkit._gdmd.gdmd import differ, tf_spec_from_gd


class AGNCMDTest(unittest.TestCase):
    """Tests for every public helper and AGNCMD method."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = make_agncmd_demo_signal(samp_freq=100.0, duration=10.0)
        cls.signal = cls.demo["signal"]
        cls.fs = float(cls.demo["fs"][0])
        cls.t = cls.demo["t"]
        cls.f = cls.demo["f"]
        cls.true_gds = cls.demo["true_gds"]
        cls.true_modes_t = cls.demo["true_modes_time"]
        cls.true_modes_f = cls.demo["true_modes_freq"]

    # ------------------------------------------------------------------ demo
    def test_make_agncmd_demo_signal(self) -> None:
        self.assertEqual(self.signal.ndim, 1)
        self.assertEqual(self.signal.size, 1000)
        self.assertEqual(self.true_gds.shape, (3, 501))
        self.assertEqual(self.true_modes_t.shape, (3, 1000))
        self.assertEqual(self.true_modes_f.shape, (3, 501))
        self.assertTrue(np.allclose(self.signal, self.true_modes_t.sum(0), atol=1e-10))

    # ------------------------------------------------------------------ helpers
    def test_differ_complex(self) -> None:
        t = np.linspace(0.0, 1.0, 101)
        y = t**2 + 1j * t
        dy = differ_complex(y, t[1] - t[0])
        self.assertEqual(dy.shape, y.shape)
        # Interior ≈ 2t + 1j
        mid = slice(10, 90)
        self.assertTrue(np.allclose(np.real(dy[mid]), 2 * t[mid], atol=0.05))
        self.assertTrue(np.allclose(np.imag(dy[mid]), 1.0, atol=0.05))
        # Real path matches float differ
        self.assertTrue(
            np.allclose(differ_complex(t**2, t[1] - t[0]), differ(t**2, t[1] - t[0]))
        )

    def test_findev(self) -> None:
        t = np.linspace(0, 4 * np.pi, 400)
        s = np.sin(t)
        vals, idx, up = findev(s)
        self.assertGreater(idx.size, 0)
        self.assertEqual(vals.size, idx.size)
        self.assertEqual(up.size, s.size)
        self.assertTrue(np.all(np.isfinite(up)))
        # Empty / short
        v, i, u = findev(np.array([1.0, 2.0]))
        self.assertEqual(u.size, 2)

    def test_arccos_phase(self) -> None:
        g = np.cos(np.linspace(0, 2 * np.pi, 100))
        th = arccos_phase(g)
        self.assertEqual(th.shape, g.shape)
        self.assertTrue(np.all(np.isfinite(th)))
        # Monotonic trend overall for a full cosine wave unwrapping
        self.assertGreater(float(th[-1]), float(th[0]))
        # Empty
        self.assertEqual(arccos_phase(np.array([])).size, 0)

    def test_spectrum_to_time_agdmd(self) -> None:
        nt = 1000
        nf = nt // 2 + 1
        # Impulse-like flat spectrum → peak near t=0 after ifft
        S = np.ones(nf, dtype=complex)
        x = spectrum_to_time_agdmd(S, nt)
        self.assertEqual(x.size, nt)
        self.assertTrue(np.isrealobj(x))
        with self.assertRaises(ValueError):
            spectrum_to_time_agdmd(np.ones(10), nt)

    def test_low_filter(self) -> None:
        fs = 100.0
        t = np.arange(0, 2.0, 1 / fs)
        x = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
        y = low_filter(x, cut_freq=8.0, samp_freq=fs)
        self.assertEqual(y.size, x.size)
        # Low-frequency energy should dominate after filtering
        Y = np.abs(np.fft.rfft(np.real(y)))
        freqs = np.fft.rfftfreq(y.size, 1 / fs)
        self.assertGreater(float(Y[freqs < 5].sum()), float(Y[freqs > 20].sum()))

    def test_tvlp(self) -> None:
        fs = 100.0
        n = 200
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 10 * t)
        eif = 10 * np.ones(n)
        y = tvlp(x, fs, eif, c_pass=5.0)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))
        with self.assertRaises(ValueError):
            tvlp(x, fs, eif[:10], 5.0)

    def test_if_dn(self) -> None:
        # Pure tone → nearly constant IF near carrier
        fs = 200.0
        t = np.arange(0, 1.0, 1 / fs)
        x = np.cos(2 * np.pi * 20 * t)
        inst = if_dn(x, fs, beta=1e-4)
        self.assertEqual(inst.size, x.size)
        self.assertTrue(np.all(np.isfinite(inst)))
        # Complex spectrum path (AGDI style)
        X = np.fft.fft(x)[: x.size // 2 + 1]
        duration = x.size / fs
        gd = if_dn(X, duration, 1e-4)
        self.assertEqual(gd.size, X.size)
        self.assertTrue(np.all(np.isfinite(gd)))

    def test_ddgdi_and_agdi(self) -> None:
        igd = agdi(self.signal, self.fs, beta=1e-7)
        self.assertEqual(igd.shape, (self.f.size,))
        self.assertTrue(np.all(np.isfinite(igd)))
        # Shape of AGDI GD should correlate with the true (parallel) GDs
        corr = abs(float(np.corrcoef(igd, self.true_gds[0])[0, 1]))
        self.assertGreater(corr, 0.9)

        # Direct DDGDI on extended spectrum
        n = self.signal.size
        x_ext = np.concatenate([self.signal, np.zeros(n), np.zeros(n)])
        t2 = x_ext.size / self.fs
        dsn = np.fft.fft(x_ext)[: x_ext.size // 2 + 1]
        ini = ddgdi(dsn, t2, 1e-7, max_iter=5)
        self.assertEqual(ini.size, dsn.size)
        self.assertTrue(np.all(np.isfinite(ini)))

    def test_dispersion_compensation_and_be(self) -> None:
        igd = agdi(self.signal, self.fs, 1e-7)
        sig_d, dsn_d = dispersion_compensation(self.signal, self.fs, igd)
        self.assertEqual(sig_d.size, self.signal.size)
        self.assertGreater(dsn_d.size, 0)
        self.assertTrue(np.all(np.isfinite(sig_d)))

        bw, alpha = bandwidth_estimation(self.signal, self.fs, igd)
        self.assertGreater(bw, 0.0)
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 10.0)
        self.assertGreaterEqual(alpha, 1e-4)

    def test_agdmd_core(self) -> None:
        nf = self.f.size
        duration = self.signal.size / self.fs
        # Good init → mode recovery
        init = self.true_gds[0:1]
        _, alpha0 = bandwidth_estimation(self.signal, self.fs, init[0])
        egd, mode_f, alpha_hist = agdmd_core(
            np.fft.fft(self.signal)[:nf],
            duration,
            init,
            alpha0,
            1e-7,
            self.signal,
            self.fs,
            max_iter=80,
        )
        self.assertEqual(egd.shape, (nf,))
        self.assertEqual(mode_f.shape, (nf,))
        self.assertGreaterEqual(alpha_hist.size, 1)
        mode_t = spectrum_to_time_agdmd(mode_f, self.signal.size)
        corr = abs(float(np.corrcoef(self.true_modes_t[0], mode_t)[0, 1]))
        self.assertGreater(corr, 0.9)
        self.assertLess(float(np.mean(np.abs(egd - self.true_gds[0]))), 0.2)

    def test_stft_agncmd(self) -> None:
        spec, f = stft_agncmd(self.signal, self.fs, n_freq=256, win_len=64)
        self.assertEqual(spec.shape[1], self.signal.size)
        self.assertEqual(f.ndim, 1)
        self.assertTrue(np.all(spec >= 0))

    def test_tf_spec_from_gd(self) -> None:
        a_spec, t_bins = tf_spec_from_gd(
            self.true_gds, np.abs(self.true_modes_f), (0.0, 10.0), n_time_bins=128
        )
        self.assertEqual(a_spec.shape[0], self.true_gds.shape[1])
        self.assertEqual(t_bins.size, 128)

    # ------------------------------------------------------------------ class
    def test_str_and_alias(self) -> None:
        self.assertIn("AGDMD", str(AGNCMD()))
        self.assertIs(AGDMD, AGNCMD)
        self.assertIs(agdmd, agncmd)

    def test_short_signal_raises(self) -> None:
        with self.assertRaises(ValueError):
            AGNCMD().fit_transform(np.ones(8), fs=10.0)
        with self.assertRaises(ValueError):
            AGNCMD().fit_transform(self.signal, fs=-1.0)
        with self.assertRaises(ValueError):
            AGNCMD(max_modes=0).fit_transform(self.signal, fs=self.fs)

    def test_fit_transform_shapes(self) -> None:
        decomp = AGNCMD(beta=1e-7, max_modes=3, max_iter=60)
        modes = decomp.fit_transform(self.signal, self.fs)
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.signal.size)
        self.assertGreaterEqual(modes.shape[0], 1)
        self.assertLessEqual(modes.shape[0], 3)
        self.assertIsNotNone(decomp.modes_time_)
        self.assertIsNotNone(decomp.group_delays_)
        self.assertIsNotNone(decomp.modes_freq_)
        self.assertIsNotNone(decomp.init_gds_)
        self.assertIsNotNone(decomp.alphas_)
        self.assertIsNotNone(decomp.residual_)
        self.assertIsNotNone(decomp.freq_)
        # Perfect reconstruction by construction
        self.assertTrue(
            np.allclose(modes.sum(0) + decomp.residual_, self.signal, atol=1e-8)
        )

    def test_return_all_and_call(self) -> None:
        decomp = AGNCMD(beta=1e-7, max_modes=2, max_iter=40)
        out = decomp(self.signal, self.fs, return_all=True)
        self.assertEqual(len(out), 6)
        modes_t, freq, igd, egd, modes_f, alphas = out
        self.assertEqual(modes_t.shape[1], self.signal.size)
        self.assertEqual(freq.size, self.signal.size // 2 + 1)
        self.assertEqual(igd.shape[0], modes_t.shape[0])
        self.assertEqual(egd.shape, igd.shape)
        self.assertEqual(modes_f.shape, egd.shape)
        self.assertEqual(len(alphas), modes_t.shape[0])
        # __call__ without return_all
        modes2 = decomp(self.signal, self.fs)
        self.assertEqual(modes2.shape, modes_t.shape)

    def test_functional_interface(self) -> None:
        modes = agncmd(self.signal, self.fs, max_modes=2, max_iter=40)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_demo_recovers_group_delays(self) -> None:
        """MATLAB Example1: estimated GDs should match the three true curves."""
        decomp = AGNCMD(beta=1e-7, max_modes=3, max_iter=120)
        modes, freq, igd, egd, modes_f, alphas = decomp.fit_transform(
            self.signal, self.fs, return_all=True
        )
        self.assertEqual(egd.shape[0], 3)
        # Each estimated GD matches some true GD with high correlation
        for k in range(3):
            corrs = [
                abs(float(np.corrcoef(egd[k], self.true_gds[j])[0, 1]))
                for j in range(3)
            ]
            self.assertGreater(max(corrs), 0.90)
        # At least one time-domain mode is recovered accurately
        best_mode = 0.0
        for k in range(modes.shape[0]):
            for j in range(3):
                best_mode = max(
                    best_mode,
                    abs(float(np.corrcoef(modes[k], self.true_modes_t[j])[0, 1])),
                )
        self.assertGreater(best_mode, 0.95)
        # Alpha histories are non-increasing (adaptive rule)
        for a in alphas:
            if a.size > 1:
                self.assertTrue(np.all(a[1:] <= a[:-1] + 1e-9))


if __name__ == "__main__":
    unittest.main()
