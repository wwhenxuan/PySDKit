# -*- coding: utf-8 -*-
"""
Created on 2025/07/23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

import numpy as np
from scipy.signal.windows import hann

from pysdkit import SWD, swd
from pysdkit._osd.swd import (
    swf_params_from_frequency,
    bandpass_swf,
    iterative_swf,
)


def _paper_atoms(n: int = 499) -> tuple:
    """Synthetic atoms from the SWD paper / EvaluationSignalGenerator demo."""

    def atom(t0, f0, T, A):
        win = np.zeros(n)
        a = max(0, int(np.ceil(t0 - T / 2 + 1)) - 1)
        b = min(n, int(np.ceil(t0 + T / 2)))
        w = hann(max(b - a, 1), sym=False)
        win[a : a + len(w)] = w[: len(win[a:b])]
        return A * win * np.cos(f0 * (np.arange(n) - t0))

    a1 = atom(250, 0.2 * np.pi, 300, 0.7)
    a2 = atom(125, 0.6 * np.pi, 125, 1.5)
    a3 = atom(375, 0.7 * np.pi, 100, 1.9)
    return a1 + a2 + a3, (a1, a2, a3)


class SWDTest(unittest.TestCase):
    """Unit tests for Swarm Decomposition."""

    signal, atoms = _paper_atoms(499)

    def test_fit_transform_shape(self) -> None:
        decomp = SWD(P_th=0.05, StD_th=0.05, max_components=6, refine=False)
        modes = decomp.fit_transform(self.signal)
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.signal.size)
        self.assertGreaterEqual(modes.shape[0], 2)

    def test_default_call(self) -> None:
        decomp = SWD(P_th=0.05, StD_th=0.05, max_components=6, refine=False)
        a = decomp(self.signal)
        b = decomp.fit_transform(self.signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_return_all(self) -> None:
        decomp = SWD(P_th=0.05, StD_th=0.05, max_components=6, refine=False)
        modes, residue, omegas = decomp.fit_transform(self.signal, return_all=True)
        self.assertEqual(modes.shape[0], omegas.size)
        self.assertEqual(residue.shape, self.signal.shape)
        recon = modes.sum(axis=0) + residue
        self.assertTrue(np.allclose(recon, self.signal, atol=1e-8))

    def test_functional_interface(self) -> None:
        modes = swd(self.signal, P_th=0.05, StD_th=0.05, max_components=6, refine=False)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_recovers_three_bands(self) -> None:
        """Paper-style three-atom signal → about three oscillatory modes."""
        decomp = SWD(P_th=0.05, StD_th=0.05, max_components=6, refine=True)
        modes, _, omegas = decomp.fit_transform(self.signal, return_all=True)
        self.assertGreaterEqual(modes.shape[0], 3)
        # Expected normalised frequencies ≈ 0.2, 0.6, 0.7
        expected = np.array([0.2, 0.6, 0.7])
        for e in expected:
            self.assertTrue(
                np.any(np.abs(omegas - e) < 0.08),
                msg=f"missing band near {e}, got {omegas}",
            )

    def test_swf_param_map(self) -> None:
        """GA-fitted M(ω̂), δ(ω̂) should be monotone in the expected direction."""
        m_lo, d_lo = swf_params_from_frequency(0.2)
        m_hi, d_hi = swf_params_from_frequency(0.8)
        self.assertGreater(m_lo, m_hi)  # slower modes need a larger swarm
        self.assertGreater(d_hi, d_lo)  # higher ω̂ → larger δ (Fig. 2)

    def test_bandpass_swf_energy(self) -> None:
        """Spectral SwF must keep a non-trivial fraction of signal energy."""
        y = bandpass_swf(self.signal, omega_hat=0.7, delta=1.0)
        self.assertGreater(np.sum(y**2), 0.05 * np.sum(self.signal**2))

    def test_iterative_swf_finite(self) -> None:
        y = iterative_swf(
            self.signal, omega_hat=0.6, std_th=0.05, max_sift=8, refine=False
        )
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertEqual(y.shape, self.signal.shape)

    def test_wrong_spectrum(self) -> None:
        with self.assertRaises(ValueError):
            SWD(spectrum="invalid")

    def test_str(self) -> None:
        self.assertIn("SWD", str(SWD()))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            SWD().fit_transform(np.ones(4))

    def test_max_components_cap(self) -> None:
        decomp = SWD(P_th=0.01, StD_th=0.05, max_components=2, refine=False)
        modes = decomp.fit_transform(self.signal)
        self.assertLessEqual(modes.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
