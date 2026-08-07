# -*- coding: utf-8 -*-
"""
Unit tests for Variational Generalized Nonlinear Mode Decomposition.
"""

import unittest

import numpy as np

from pysdkit import VGNMD, vgnmd
from pysdkit._gdmd.vgnmd import (
    acmd_single,
    atffc,
    make_vgnmd_demo_signal,
    mtdc,
    stft_vgnmd,
    voa,
)


class VGNMDTest(unittest.TestCase):
    """Unit tests for Variational Generalized Nonlinear Mode Decomposition."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = make_vgnmd_demo_signal(samp_freq=1000.0, noise_std=0.0)
        cls.fs = float(cls.demo["fs"][0])
        cls.signal = cls.demo["signal"]
        cls.t = cls.demo["t"]
        cls.true_modes = cls.demo["modes_true"]

    def test_demo_signal_shape(self) -> None:
        self.assertEqual(self.signal.ndim, 1)
        self.assertEqual(self.true_modes.shape[0], 7)
        self.assertEqual(self.true_modes.shape[1], self.signal.size)

    def test_stft_shape(self) -> None:
        spec, f = stft_vgnmd(self.signal, self.fs, win_len=250.0)
        self.assertEqual(spec.ndim, 2)
        self.assertEqual(spec.shape[1], self.signal.size)
        self.assertEqual(f.size, spec.shape[0])

    def test_mtdc_on_atffc_clusters(self) -> None:
        """MTDC on ATFFC clusters of the Test.m mixture yields both types."""
        clusters, _ = atffc(self.signal, self.fs, n_windows=5)
        types = []
        for clu in clusters:
            mtype, idx = mtdc(clu)
            types.append(mtype)
            self.assertIn(mtype, (1, 2))
            self.assertEqual(idx.ndim, 2)
            self.assertEqual(idx.shape[1], 2)
            self.assertGreaterEqual(idx.shape[0], 1)
        self.assertIn(1, types)
        self.assertIn(2, types)

    def test_acmd_recovers_chirp(self) -> None:
        sig1 = self.true_modes[0]
        mode, if_est, ia = acmd_single(
            sig1,
            self.t,
            self.demo["if1"],
            alpha=1e-4,
            beta=1e-7,
            tol=1e-6,
            max_iter=60,
        )
        corr = np.corrcoef(mode, sig1)[0, 1]
        self.assertGreater(corr, 0.99)
        self.assertEqual(if_est.shape, sig1.shape)
        self.assertTrue(np.all(np.isfinite(ia)))

    def test_voa_chirp_branch(self) -> None:
        sig1 = self.true_modes[0]
        # synthetic ridge: full time support, IF mapped to frequency bins
        f = np.linspace(0.0, self.fs / 2.0, self.signal.size // 2)
        indexf = np.clip(
            np.searchsorted(f, np.clip(self.demo["if1"], 0, self.fs / 2)),
            0,
            f.size - 1,
        )
        ridge = np.column_stack([np.arange(sig1.size), indexf])
        mt, mf, feat, mtype = voa(
            sig1,
            1,
            ridge,
            self.t,
            f,
            alpha=1e-4,
            beta=1e-7,
            tol=1e-6,
            max_iter=40,
        )
        self.assertEqual(mtype, 1)
        self.assertGreater(np.corrcoef(mt, sig1)[0, 1], 0.95)
        self.assertEqual(mf.size, f.size)
        self.assertEqual(feat.size, sig1.size)

    def test_atffc_finds_clusters(self) -> None:
        clusters, f = atffc(self.signal, self.fs, n_windows=5)
        self.assertGreaterEqual(len(clusters), 3)
        self.assertTrue(f.size > 0)
        for c in clusters:
            self.assertEqual(c.shape[1], self.signal.size)

    def test_fit_transform_shape_and_types(self) -> None:
        decomp = VGNMD(alpha=1e-4, beta=1e-7, tol=1e-6, max_iter=60)
        modes, modes_f, types, ridges, feats, f, t = decomp.fit_transform(
            self.signal, self.fs, return_all=True
        )
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.signal.size)
        self.assertEqual(modes.shape[0], types.size)
        self.assertEqual(modes_f.shape[0], modes.shape[0])
        self.assertTrue(set(np.unique(types)).issubset({1, 2}))
        self.assertEqual(len(ridges), modes.shape[0])
        self.assertEqual(len(feats), modes.shape[0])
        self.assertEqual(t.size, self.signal.size)

    def test_recovers_seven_modes(self) -> None:
        """Paper Test.m mixture: expect ~7 modes, both chirp and dispersive."""
        decomp = VGNMD(alpha=1e-4, beta=1e-7, tol=1e-6, max_iter=80)
        modes, _, types, *_ = decomp.fit_transform(
            self.signal, self.fs, return_all=True
        )
        self.assertGreaterEqual(modes.shape[0], 5)
        self.assertTrue(np.any(types == 1))
        self.assertTrue(np.any(types == 2))

        # each true mode should match some estimate with high correlation
        for i, tr in enumerate(self.true_modes):
            corrs = [
                abs(np.corrcoef(modes[j], tr)[0, 1]) for j in range(modes.shape[0])
            ]
            self.assertGreater(
                max(corrs),
                0.85,
                msg=f"true mode {i + 1} not recovered well; max|corr|={max(corrs):.3f}",
            )

    def test_functional_interface(self) -> None:
        modes = vgnmd(self.signal, self.fs, tol=1e-6, max_iter=40)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_default_call(self) -> None:
        decomp = VGNMD(tol=1e-6, max_iter=40)
        a = decomp(self.signal, self.fs)
        b = decomp.fit_transform(self.signal, self.fs)
        self.assertEqual(a.shape, b.shape)

    def test_str(self) -> None:
        self.assertIn("VGNMD", str(VGNMD()))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            VGNMD().fit_transform(np.ones(8), fs=100.0)


if __name__ == "__main__":
    unittest.main()
