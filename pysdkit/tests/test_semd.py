# -*- coding: utf-8 -*-
"""
Unit tests for Serial Empirical Mode Decomposition (SEMD).
"""

import unittest

import numpy as np

from pysdkit import SEMD, EMD
from pysdkit._emd.semd import (
    concatenate_signals,
    deconcatenate_imfs,
    transition_bridge,
)
from pysdkit.data import test_emd, test_multivariate_signal


class SEMDTest(unittest.TestCase):
    """Automated tests for Serial-EMD."""

    def test_concatenate_length(self) -> None:
        """Serialized length must be M*N + D*(N-1)."""
        m, n, d = 100, 5, 20
        x = np.random.default_rng(0).standard_normal((m, n))
        ser = concatenate_signals(x, d)
        self.assertEqual(ser.shape, (m * n + d * (n - 1),))

    def test_concatenate_deconcatenate_roundtrip(self) -> None:
        """Identity IMFs must recover the original multi-channel block."""
        m, n, d = 80, 4, 15
        x = np.arange(m * n, dtype=float).reshape(m, n)
        ser = concatenate_signals(x, d)
        imfs = deconcatenate_imfs(ser.reshape(-1, 1), d, n, num_length=m)
        self.assertEqual(imfs.shape, (m, 1, n))
        self.assertTrue(np.allclose(imfs[:, 0, :], x))

    def test_matches_reference_concatenate(self) -> None:
        """Stay bit-compatible with the official serial-emd concatenate."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal((60, 3))
        d = 12
        a = np.linspace(0, 1, d + 2)[1:-1].reshape(-1, 1)
        u = np.ones((2, 1))
        t = np.flipud(x[:d, 1:]) * (a @ u.T) + np.flipud(x[-d:, :-1]) * (
            np.flipud(a) @ u.T
        )
        t = np.concatenate([t, np.zeros((d, 1))], axis=1)
        ref = np.concatenate([x, t], axis=0).flatten(order="F")[:-d]
        self.assertTrue(np.allclose(concatenate_signals(x, d), ref))

    def test_transition_bridge_shape(self) -> None:
        tail = np.linspace(1, 2, 30)
        head = np.linspace(-1, 0, 30)
        bridge = transition_bridge(tail, head, num_interval=10)
        self.assertEqual(bridge.shape, (10,))
        self.assertTrue(np.all(np.isfinite(bridge)))

    def test_fit_transform_univariate(self) -> None:
        """Univariate input falls back to ordinary EMD layout (K, L)."""
        _, signal = test_emd()
        imfs = SEMD(max_imfs=3).fit_transform(signal, max_imfs=3)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], signal.size)
        self.assertLessEqual(imfs.shape[0], 4)
        self.assertTrue(np.allclose(imfs.sum(0), signal, atol=1e-5))

    def test_fit_transform_multivariate_shape(self) -> None:
        """Multivariate output layout is (K, seq_len, n_channels)."""
        _, signal = test_multivariate_signal(case=1)
        n_ch, seq_len = signal.shape
        imfs = SEMD(num_interval=50, max_imfs=4).fit_transform(signal, max_imfs=4)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1], seq_len)
        self.assertEqual(imfs.shape[2], n_ch)
        self.assertLessEqual(imfs.shape[0], 5)

    def test_reconstruction_multivariate(self) -> None:
        """Summing IMFs along K recovers each original channel."""
        _, signal = test_multivariate_signal(case=1)
        semd = SEMD(num_interval=40, max_imfs=6)
        imfs = semd.fit_transform(signal)
        recon = semd.reconstruct(imfs)
        self.assertEqual(recon.shape, signal.shape)
        self.assertTrue(np.allclose(recon, signal, atol=1e-4))

    def test_default_call(self) -> None:
        _, signal = test_multivariate_signal(case=1)
        semd = SEMD(num_interval=30, max_imfs=3)
        a = semd(signal)
        b = semd.fit_transform(signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_str(self) -> None:
        self.assertIn("SEMD", str(SEMD()))

    def test_interval_ratio_default(self) -> None:
        """Default D ≈ 0.2 * M when num_interval is omitted."""
        seq_len = 200
        signal = np.random.default_rng(0).standard_normal((3, seq_len))
        semd = SEMD(interval_ratio=0.2, max_imfs=2)
        semd.fit_transform(signal)
        self.assertEqual(semd._last_D, 40)

    def test_custom_emd_backend(self) -> None:
        """Accept a pre-configured univariate EMD instance."""
        _, signal = test_multivariate_signal(case=1)
        backend = EMD(max_imfs=2, max_iteration=200)
        imfs = SEMD(num_interval=25, emd=backend).fit_transform(signal)
        self.assertEqual(imfs.shape[2], signal.shape[0])
        self.assertLessEqual(imfs.shape[0], 3)

    def test_invalid_interval(self) -> None:
        with self.assertRaises(ValueError):
            SEMD(num_interval=0)
        x = np.ones((50, 3))
        with self.assertRaises(ValueError):
            concatenate_signals(x, 50)

    def test_single_channel_matrix(self) -> None:
        """N=1 concatenate is a no-op copy of the column."""
        col = np.sin(np.linspace(0, 4 * np.pi, 100))
        ser = concatenate_signals(col.reshape(-1, 1), 10)
        self.assertTrue(np.allclose(ser, col))

    def test_naive_concat_vs_bridge(self) -> None:
        """Bridged serialization should be closer to C0 at joins than hard joins."""
        m, d = 100, 20
        t = np.linspace(0, 1, m)
        ch0 = np.sin(2 * np.pi * 3 * t)
        ch1 = np.cos(2 * np.pi * 5 * t) + 2.0  # level shift
        x = np.column_stack([ch0, ch1])
        ser = concatenate_signals(x, d)
        # Hard join discontinuity magnitude
        hard_gap = abs(ch0[-1] - ch1[0])
        # Bridged join: sample at end of ch0 vs first transition sample
        soft_gap = abs(ser[m] - ser[m - 1])
        self.assertLess(soft_gap, hard_gap)


if __name__ == "__main__":
    unittest.main()
