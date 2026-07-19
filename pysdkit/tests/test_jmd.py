# -*- coding: utf-8 -*-
"""
Unit tests for JMD and MJMD.
"""
import unittest
import numpy as np

from pysdkit import JMD, MJMD


def _make_jmd_signal(n: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) / n
    s1 = 2.0 * np.cos(2 * np.pi * 8 * t)
    s2 = 1.5 * np.cos(2 * np.pi * 40 * t)
    jump = np.zeros(n)
    jump[n // 3 :] += 2.0
    jump[2 * n // 3 :] -= 1.5
    signal = s1 + s2 + jump + 0.05 * rng.standard_normal(n)
    return t, signal, s1, s2, jump


def _make_mjmd_signal(n: int = 256, c: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) / n
    x = np.vstack(
        [
            3.0 * np.cos(2 * np.pi * 2 * t) + np.cos(2 * np.pi * 20 * t),
            2.0 * np.cos(2 * np.pi * 2 * t) + np.cos(2 * np.pi * 30 * t),
            2.0 * np.cos(2 * np.pi * 20 * t) + np.cos(2 * np.pi * 30 * t),
        ]
    )
    jump = np.zeros((c, n))
    jump[:, n // 2 :] += 1.5
    signal = x + jump + 0.05 * rng.standard_normal((c, n))
    return t, signal, x, jump


class JMDTest(unittest.TestCase):
    def test_shapes(self) -> None:
        _, signal, _, _, _ = _make_jmd_signal()
        jmd = JMD(K=2, alpha=5000, tau=5, beta=0.03, max_iter=120, tol=1e-4)
        u, v, omega = jmd.fit_transform(signal, return_all=True)
        self.assertEqual(u.shape, (2, signal.size))
        self.assertEqual(v.shape, (signal.size,))
        self.assertEqual(omega.shape, (2,))

    def test_call(self) -> None:
        _, signal, _, _, _ = _make_jmd_signal(n=256)
        jmd = JMD(K=2, max_iter=80, tol=1e-4)
        self.assertEqual(jmd(signal).shape, jmd.fit_transform(signal).shape)

    def test_reconstruction(self) -> None:
        _, signal, _, _, _ = _make_jmd_signal()
        jmd = JMD(K=2, alpha=5000, tau=5, beta=0.03, max_iter=250, tol=1e-5)
        u, v, _ = jmd.fit_transform(signal, return_all=True)
        recon = u.sum(axis=0) + v
        re = np.linalg.norm(recon - signal) / np.linalg.norm(signal)
        self.assertLess(re, 0.15)

    def test_omega_sorted(self) -> None:
        _, signal, _, _, _ = _make_jmd_signal(n=256)
        jmd = JMD(K=2, max_iter=100, tol=1e-4)
        _, _, omega = jmd.fit_transform(signal, return_all=True)
        self.assertTrue(np.all(np.diff(omega) >= -1e-12))

    def test_str(self) -> None:
        self.assertIn("JMD", str(JMD(K=2)))


class MJMDTest(unittest.TestCase):
    def test_shapes(self) -> None:
        _, signal, _, _ = _make_mjmd_signal()
        mjmd = MJMD(K=2, alpha=3000, tau=5, beta=0.05, max_iter=100, tol=1e-4)
        u, jump, omega = mjmd.fit_transform(signal, return_all=True)
        self.assertEqual(u.shape, (2, signal.shape[1], signal.shape[0]))
        self.assertEqual(jump.shape, signal.shape)
        self.assertEqual(omega.shape, (2,))

    def test_accepts_time_major(self) -> None:
        """MATLAB-style (N, C) input should be accepted when N > C."""
        _, signal, _, _ = _make_mjmd_signal(n=200, c=3)
        mjmd = MJMD(K=2, max_iter=60, tol=1e-4)
        u = mjmd.fit_transform(signal.T)  # (N, C)
        self.assertEqual(u.shape[0], 2)
        self.assertEqual(u.shape[1], 200)
        self.assertEqual(u.shape[2], 3)

    def test_reconstruction(self) -> None:
        _, signal, _, _ = _make_mjmd_signal()
        mjmd = MJMD(K=2, alpha=3000, tau=5, beta=0.05, max_iter=180, tol=1e-4)
        u, jump, _ = mjmd.fit_transform(signal, return_all=True)
        recon = u.sum(axis=0).T + jump
        re = np.linalg.norm(recon - signal) / np.linalg.norm(signal)
        self.assertLess(re, 0.25)

    def test_omega_sorted(self) -> None:
        _, signal, _, _ = _make_mjmd_signal()
        mjmd = MJMD(K=2, max_iter=80, tol=1e-4)
        _, _, omega = mjmd.fit_transform(signal, return_all=True)
        self.assertTrue(np.all(np.diff(omega) >= -1e-12))

    def test_str(self) -> None:
        self.assertIn("MJMD", str(MJMD(K=2)))


if __name__ == "__main__":
    unittest.main()
