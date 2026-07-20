# -*- coding: utf-8 -*-
"""
Unit tests for SJMD / SMJMD.
"""
import unittest
import numpy as np

from pysdkit import SJMD, SMJMD


def _make_uni_signal(n: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) / n
    s1 = 4.0 * np.cos(2 * np.pi * 2 * t)
    s2 = 2.0 * np.cos(2 * np.pi * 40 * t)
    jump = np.zeros(n)
    jump[n // 3 :] += 2.0
    jump[2 * n // 3 :] -= 1.5
    signal = s1 + s2 + jump + 0.05 * rng.standard_normal(n)
    return t, signal, s1, s2, jump


def _make_multi_signal(n: int = 200, seed: int = 0):
    """Synthetic 3-channel signal following ``SJMD_test.m`` structure."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) / n
    jump = np.zeros(n)
    jump[n // 4 :] += 1.0
    jump[n // 2 :] -= 0.5
    jump[3 * n // 4 :] += 0.8
    jump_mc = np.vstack([jump, np.zeros(n), jump])

    ch1 = 5.0 * np.cos(2 * np.pi * 2 * t) + 2.0 * np.cos(2 * np.pi * 40 * t)
    ch2 = 5.0 * np.cos(2 * np.pi * 2 * t) + 2.0 * np.cos(2 * np.pi * 40 * t)
    ch3 = 5.0 * np.cos(2 * np.pi * 2 * t)
    osc = np.vstack([ch1, ch2, ch3])
    signal = osc + jump_mc + 0.05 * rng.standard_normal((3, n))
    return t, signal, osc, jump_mc


class SJMDUnivariateTest(unittest.TestCase):
    def test_shapes(self) -> None:
        _, signal, _, _, _ = _make_uni_signal()
        sjmd = SJMD(
            max_alpha=5000,
            tau=20,
            beta=0.5,
            b_bar=0.9,
            stopc=3,
            kdm=2,
            tol=1e-4,
            max_inner=80,
        )
        u, v = sjmd.fit_transform(signal, return_all=True)
        self.assertEqual(u.shape, (2, signal.size))
        self.assertEqual(v.shape, (signal.size,))

    def test_call(self) -> None:
        _, signal, _, _, _ = _make_uni_signal(n=128)
        sjmd = SJMD(max_alpha=2000, stopc=3, kdm=1, max_inner=40, tol=1e-3)
        self.assertEqual(sjmd(signal).shape, sjmd.fit_transform(signal).shape)

    def test_reconstruction(self) -> None:
        _, signal, _, _, _ = _make_uni_signal(n=256)
        sjmd = SJMD(
            max_alpha=20000,
            tau=50,
            beta=0.5,
            b_bar=0.9,
            stopc=3,
            kdm=2,
            tol=1e-4,
            max_inner=150,
        )
        u, v = sjmd.fit_transform(signal, return_all=True)
        recon = u.sum(axis=0) + v
        re = np.linalg.norm(recon - signal) / np.linalg.norm(signal)
        # Successive extraction is approximate (unlike joint JMD)
        self.assertLess(re, 0.55)

    def test_recovers_jump_and_low_mode(self) -> None:
        _, signal, s1, _, jump = _make_uni_signal(n=256)
        sjmd = SJMD(
            max_alpha=20000,
            tau=50,
            beta=0.5,
            b_bar=0.9,
            stopc=3,
            kdm=2,
            tol=1e-4,
            max_inner=150,
        )
        u, v = sjmd.fit_transform(signal, return_all=True)
        # lowest-frequency mode should track the slow cosine
        self.assertGreater(abs(np.corrcoef(u[0], s1)[0, 1]), 0.9)
        self.assertGreater(abs(np.corrcoef(v, jump)[0, 1]), 0.85)

    def test_omega_sorted(self) -> None:
        _, signal, _, _, _ = _make_uni_signal(n=128)
        sjmd = SJMD(max_alpha=3000, stopc=3, kdm=2, max_inner=60, tol=1e-3)
        sjmd.fit_transform(signal)
        self.assertIsNotNone(sjmd.omega)
        self.assertTrue(np.all(np.diff(sjmd.omega) >= -1e-12))

    def test_str(self) -> None:
        self.assertIn("SJMD", str(SJMD()))


class SJMDMultivariateTest(unittest.TestCase):
    def test_shapes(self) -> None:
        _, signal, _, _ = _make_multi_signal()
        sjmd = SJMD(
            max_alpha=5000,
            tau=20,
            beta=0.5,
            b_bar=0.9,
            stopc=3,
            kdm=2,
            tol=1e-4,
            max_inner=60,
        )
        u, jump = sjmd.fit_transform(signal, return_all=True)
        self.assertEqual(u.shape, (2, signal.shape[1], signal.shape[0]))
        self.assertEqual(jump.shape, signal.shape)

    def test_accepts_time_major(self) -> None:
        """MATLAB-style (N, C) input should be accepted when N > C."""
        _, signal, _, _ = _make_multi_signal(n=160)
        sjmd = SJMD(max_alpha=3000, stopc=3, kdm=2, max_inner=40, tol=1e-3)
        u = sjmd.fit_transform(signal.T)
        self.assertEqual(u.shape[0], 2)
        self.assertEqual(u.shape[1], 160)
        self.assertEqual(u.shape[2], 3)

    def test_reconstruction(self) -> None:
        _, signal, _, _ = _make_multi_signal()
        sjmd = SJMD(
            max_alpha=20000,
            tau=50,
            beta=0.5,
            b_bar=0.9,
            stopc=3,
            kdm=2,
            tol=1e-4,
            max_inner=120,
        )
        u, jump = sjmd.fit_transform(signal, return_all=True)
        recon = u.sum(axis=0).T + jump
        re = np.linalg.norm(recon - signal) / np.linalg.norm(signal)
        self.assertLess(re, 0.55)

    def test_smjmd_alias(self) -> None:
        self.assertIs(SMJMD, SJMD)

    def test_str(self) -> None:
        self.assertIn("SMJMD", str(SMJMD()))


if __name__ == "__main__":
    unittest.main()
