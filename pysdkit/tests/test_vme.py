# -*- coding: utf-8 -*-
"""
Automated tests for Variational Mode Extraction (VME).

Covers every public function and method in ``pysdkit._vmd.vme``, including
helpers ported from MATLAB ``vme.m`` and the File Exchange demos in
``VME_test_script.m``.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import VME, vme
from pysdkit._vmd.vme import (
    compactness_kernel,
    crop_mirror,
    ensure_even_length,
    generate_vme_example1,
    generate_vme_example2,
    generate_vme_example3a,
    generate_vme_example3b,
    load_vme_ecg_055m,
    mirror_extend,
    onesided_fft,
    reconstruct_hermitian,
    relative_spectrum_change,
    residual_spectrum,
    spectral_axis,
    spectrum_to_time,
    update_center_frequency,
    update_dual,
    update_mode_spectrum,
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class EnsureEvenLengthTest(unittest.TestCase):
    def test_even_passthrough(self) -> None:
        x = np.arange(8, dtype=float)
        np.testing.assert_array_equal(ensure_even_length(x), x)

    def test_odd_drops_last(self) -> None:
        x = np.arange(9, dtype=float)
        np.testing.assert_array_equal(ensure_even_length(x), x[:-1])

    def test_too_short(self) -> None:
        with self.assertRaises(ValueError):
            ensure_even_length(np.array([1.0]))


class MirrorExtendTest(unittest.TestCase):
    def test_length_and_halves(self) -> None:
        x = np.arange(10, dtype=float)
        ext = mirror_extend(x)
        self.assertEqual(ext.size, 20)
        np.testing.assert_array_equal(ext[:5], x[:5][::-1])
        np.testing.assert_array_equal(ext[5:15], x)
        np.testing.assert_array_equal(ext[15:], x[5:][::-1])

    def test_crop_inverts_extend(self) -> None:
        x = np.linspace(-1.0, 1.0, 16)
        np.testing.assert_allclose(crop_mirror(mirror_extend(x)), x)

    def test_odd_input_then_crop(self) -> None:
        x = np.arange(11, dtype=float)
        cropped = crop_mirror(mirror_extend(x))
        np.testing.assert_array_equal(cropped, x[:-1])


class SpectralHelpersTest(unittest.TestCase):
    def test_spectral_axis_matches_matlab(self) -> None:
        n = 8
        t = np.arange(1, n + 1, dtype=float) / n
        expected = t - 0.5 - 1.0 / n
        np.testing.assert_allclose(spectral_axis(n), expected)
        self.assertAlmostEqual(spectral_axis(n)[n // 2], 0.0, places=12)

    def test_onesided_zeros_negative_bins(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=32)
        spec = onesided_fft(x)
        self.assertEqual(spec.dtype, np.complex128)
        np.testing.assert_array_equal(spec[: spec.size // 2], 0.0)
        self.assertTrue(np.any(np.abs(spec[spec.size // 2 :]) > 0.0))

    def test_compactness_kernel_at_center(self) -> None:
        omega = np.array([-0.25, 0.0, 0.25])
        kernel = compactness_kernel(omega, omega_d=0.0, alpha=10.0)
        self.assertEqual(kernel.shape, omega.shape)
        self.assertAlmostEqual(float(kernel[1]), 0.0)
        self.assertTrue(np.all(kernel >= 0.0))


class UpdateStepsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 16
        self.omega = spectral_axis(self.n)
        self.f_hat = onesided_fft(np.cos(2.0 * np.pi * np.arange(self.n) / 8.0))
        self.u0 = np.zeros(self.n, dtype=np.complex128)
        self.dual0 = np.zeros(self.n, dtype=np.complex128)
        self.alpha = 50.0

    def test_mode_spectrum_finite_and_onesided(self) -> None:
        u1 = update_mode_spectrum(
            self.f_hat, self.u0, self.dual0, self.omega, 0.1, self.alpha
        )
        self.assertEqual(u1.dtype, np.complex128)
        self.assertTrue(np.all(np.isfinite(u1)))
        np.testing.assert_array_equal(u1[: self.n // 2], 0.0)

    def test_residual_zero_when_u_equals_f(self) -> None:
        residual = residual_spectrum(
            self.f_hat, self.f_hat, self.omega, 0.1, self.alpha
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_dual_unchanged_when_tau_zero(self) -> None:
        u1 = update_mode_spectrum(
            self.f_hat, self.u0, self.dual0, self.omega, 0.1, self.alpha
        )
        dual1 = update_dual(
            self.dual0, self.f_hat, u1, self.omega, 0.1, self.alpha, tau=0.0
        )
        np.testing.assert_array_equal(dual1, self.dual0)

    def test_dual_steps_when_tau_positive(self) -> None:
        u1 = update_mode_spectrum(
            self.f_hat, self.u0, self.dual0, self.omega, 0.1, self.alpha
        )
        dual1 = update_dual(
            self.dual0, self.f_hat, u1, self.omega, 0.1, self.alpha, tau=0.5
        )
        self.assertFalse(np.allclose(dual1, self.dual0))

    def test_center_frequency_in_unit_interval(self) -> None:
        u1 = update_mode_spectrum(
            self.f_hat, self.u0, self.dual0, self.omega, 0.1, self.alpha
        )
        omega = update_center_frequency(u1, self.omega, previous=0.1)
        self.assertGreaterEqual(omega, -0.5)
        self.assertLessEqual(omega, 0.5)

    def test_center_frequency_zero_power_keeps_previous(self) -> None:
        zeros = np.zeros(self.n, dtype=np.complex128)
        omega = update_center_frequency(zeros, self.omega, previous=0.17)
        self.assertAlmostEqual(omega, 0.17)

    def test_relative_change_zero_for_identical(self) -> None:
        x = np.ones(8, dtype=np.complex128)
        self.assertAlmostEqual(relative_spectrum_change(x, x), np.finfo(float).eps)


class ReconstructTest(unittest.TestCase):
    def test_hermitian_symmetry_after_overwrite(self) -> None:
        t = 8
        pos = np.zeros(t, dtype=np.complex128)
        pos[4:] = np.array([1, 2 + 1j, 3 - 2j, 4 + 0.5j])
        u_hat = reconstruct_hermitian(pos)
        self.assertEqual(u_hat.size, t)
        np.testing.assert_allclose(u_hat[0], np.conj(u_hat[-1]))
        # MATLAB overwrites Nyquist bin with conj(pos[half])
        np.testing.assert_allclose(u_hat[4], np.conj(pos[4]))

    def test_spectrum_to_time_real(self) -> None:
        x = np.cos(2.0 * np.pi * np.arange(32) / 8.0)
        spec = np.fft.fftshift(np.fft.fft(x))
        rec = spectrum_to_time(spec)
        np.testing.assert_allclose(rec, x, atol=1e-10)


class DemoGeneratorsTest(unittest.TestCase):
    def test_example_shapes(self) -> None:
        for factory in (
            generate_vme_example1,
            generate_vme_example2,
            generate_vme_example3a,
            generate_vme_example3b,
        ):
            demo = factory(n_samples=1000, fs=1000.0)
            self.assertEqual(demo["signal"].shape, (1000,))
            self.assertEqual(demo["reference"].shape, (1000,))
            self.assertEqual(demo["t"].shape, (1000,))
            self.assertEqual(demo["fs"], 1000.0)

    def test_example2_three_tones(self) -> None:
        demo = generate_vme_example2()
        self.assertGreater(np.std(demo["signal"]), np.std(demo["reference"]))


class LoadEcgTest(unittest.TestCase):
    def test_packaged_mimic_055m(self) -> None:
        demo = load_vme_ecg_055m()
        self.assertEqual(demo["val"].shape, (7, 7500))
        self.assertEqual(demo["ecg"].shape, (7500,))
        self.assertEqual(demo["respiration"].shape, (7500,))
        self.assertEqual(demo["fs"], 125.0)
        self.assertEqual(demo["t"].size, 7500)
        np.testing.assert_array_equal(demo["ecg"], demo["val"][0])
        np.testing.assert_array_equal(demo["respiration"], demo["val"][-1])
        self.assertAlmostEqual(demo["t"][-1], (7500 - 1) / 125.0)


class VMEClassTest(unittest.TestCase):
    def test_str(self) -> None:
        self.assertIn("VME", str(VME()))

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            VME(alpha=-1.0)
        with self.assertRaises(ValueError):
            VME(fs=0.0)
        with self.assertRaises(ValueError):
            VME(max_iter=1)
        with self.assertRaises(ValueError):
            VME(tol=0.0)

    def test_call_matches_fit_transform(self) -> None:
        demo = generate_vme_example2()
        a = VME(alpha=20000.0, omega_init=10.0, fs=1000.0)(demo["signal"])
        b = VME(alpha=20000.0, omega_init=10.0, fs=1000.0).fit_transform(demo["signal"])
        np.testing.assert_allclose(a, b)

    def test_return_all_shapes(self) -> None:
        demo = generate_vme_example2()
        extractor = VME(alpha=20000.0, omega_init=10.0, fs=1000.0, max_iter=80)
        u, u_hat, omega = extractor.fit_transform(demo["signal"], return_all=True)
        self.assertEqual(u.shape, demo["signal"].shape)
        self.assertEqual(u_hat.shape, demo["signal"].shape)
        self.assertGreaterEqual(omega.size, 2)
        self.assertEqual(extractor.n_iter, omega.size - 1)
        self.assertAlmostEqual(extractor.omega, float(omega[-1]))
        self.assertTrue(np.all(np.isfinite(u)))
        self.assertTrue(np.all(np.isfinite(omega)))

    def test_functional_interface(self) -> None:
        demo = generate_vme_example1()
        u_cls = VME(alpha=20000.0, omega_init=0.0, fs=1000.0).fit_transform(
            demo["signal"]
        )
        u_fn = vme(demo["signal"], alpha=20000.0, omega_init=0.0, fs=1000.0)
        np.testing.assert_allclose(u_cls, u_fn)

    def test_odd_length_truncated(self) -> None:
        demo = generate_vme_example2(n_samples=1001, fs=1000.0)
        u = VME(alpha=5000.0, omega_init=10.0, fs=1000.0, max_iter=40).fit_transform(
            demo["signal"]
        )
        self.assertEqual(u.size, 1000)

    def test_stores_attributes(self) -> None:
        demo = generate_vme_example1()
        extractor = VME(alpha=20000.0, omega_init=0.0, fs=1000.0, max_iter=50)
        u = extractor.fit_transform(demo["signal"])
        np.testing.assert_array_equal(extractor.u, u)
        self.assertIsNotNone(extractor.u_hat)
        self.assertIsNotNone(extractor.omega_hist)


class MatlabDemoAccuracyTest(unittest.TestCase):
    """Reproduce ``VME_test_script.m`` and check correlation with the target mode."""

    def test_example1_lowest_mode(self) -> None:
        demo = generate_vme_example1()
        u = VME(
            alpha=20000.0, omega_init=0.0, fs=1000.0, tau=0.0, tol=1e-7
        ).fit_transform(demo["signal"])
        self.assertGreater(_corr(u, demo["reference"]), 0.95)

    def test_example2_am_tone(self) -> None:
        demo = generate_vme_example2()
        u = VME(
            alpha=20000.0, omega_init=10.0, fs=1000.0, tau=0.0, tol=1e-7
        ).fit_transform(demo["signal"])
        self.assertGreater(_corr(u, demo["reference"]), 0.90)
        # Should not lock onto the 2 Hz trend
        c1 = 2.0 * np.cos(4.0 * np.pi * demo["t"])
        self.assertLess(_corr(u, c1), 0.4)

    def test_example3a_chirp(self) -> None:
        demo = generate_vme_example3a()
        u = VME(
            alpha=20000.0, omega_init=6.0, fs=1000.0, tau=0.0, tol=1e-7
        ).fit_transform(demo["signal"])
        self.assertGreater(_corr(u, demo["reference"]), 0.85)

    def test_example3b_piecewise_tone(self) -> None:
        demo = generate_vme_example3b()
        u = VME(
            alpha=20000.0, omega_init=26.0, fs=1000.0, tau=0.0, tol=1e-7
        ).fit_transform(demo["signal"])
        self.assertGreater(_corr(u, demo["reference"]), 0.70)

    def test_ecg_derived_respiration(self) -> None:
        rec = load_vme_ecg_055m()
        # Paper Fig. 4 uses a 32 s (4000-sample) window of record 055
        n = 4000
        extractor = VME(
            alpha=20000.0,
            omega_init=0.0,
            fs=float(rec["fs"]),
            tau=0.0,
            tol=1e-7,
        )
        u, _, omega_hist = extractor.fit_transform(rec["ecg"][:n], return_all=True)
        resp = rec["respiration"][:n]
        self.assertEqual(u.size, n)
        self.assertGreater(_corr(u, resp), 0.3)
        omega_hz = float(omega_hist[-1]) * float(rec["fs"])
        self.assertGreaterEqual(omega_hz, 0.0)
        self.assertLess(omega_hz, 1.0)


if __name__ == "__main__":
    unittest.main()
