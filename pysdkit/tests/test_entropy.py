# -*- coding: utf-8 -*-
"""
Automated tests for pysdkit.entropy.
"""

import unittest
import numpy as np

from pysdkit.entropy import (
    permutation_entropy,
    multiscale_permutation_entropy,
    sample_entropy,
    multiscale_sample_entropy,
    composite_multiscale_sample_entropy,
    refined_composite_multiscale_sample_entropy,
    approximate_entropy,
    fuzzy_entropy,
    multiscale_fuzzy_entropy,
    dispersion_entropy,
    multiscale_dispersion_entropy,
    spectral_entropy,
    distribution_entropy,
    increment_entropy,
    slope_entropy,
    symbolic_dynamic_entropy,
)


def _sine(n: int = 200, freq: float = 5.0) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.sin(2.0 * np.pi * freq * t)


def _noise(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n)


class PermutationEntropyTest(unittest.TestCase):
    """Permutation entropy and its multiscale variant."""

    def test_finite_sine(self) -> None:
        pe, hist = permutation_entropy(_sine(), m=3, t=1)
        self.assertTrue(np.isfinite(pe))
        self.assertGreater(hist.sum(), 0)

    def test_too_short_order(self) -> None:
        with self.assertRaises(ValueError):
            permutation_entropy(np.array([1.0, 2.0]), m=5, t=1)

    def test_multiscale_length(self) -> None:
        mpe = multiscale_permutation_entropy(_sine(), m=2, t=1, scale=4)
        self.assertEqual(len(mpe), 4)
        self.assertTrue(np.all(np.isfinite(mpe)))


class SampleEntropyTest(unittest.TestCase):
    """Sample entropy, Costa MSE, composite and refined-composite MSE."""

    def test_finite_sine(self) -> None:
        value = sample_entropy(_sine(), m=2, r=0.2)
        self.assertTrue(np.isfinite(value))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            sample_entropy(np.array([1.0]), m=1, r=0.2)

    def test_embedding_too_large(self) -> None:
        with self.assertRaises(ValueError):
            sample_entropy(_sine(20), m=20, r=0.2)

    def test_multiscale_tau_one(self) -> None:
        e, a_count, b_count = multiscale_sample_entropy(_sine(), m=2, r=0.15, tau=1)
        self.assertTrue(np.isfinite(e) or np.isnan(e))
        self.assertGreaterEqual(b_count, 0)
        self.assertGreaterEqual(a_count, 0)

    def test_composite_shape(self) -> None:
        rng = np.random.RandomState(1)
        values = composite_multiscale_sample_entropy(rng.randn(400), m=2, r=0.15, scale=3)
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.any(np.isfinite(values)))

    def test_refined_composite_shape(self) -> None:
        rng = np.random.RandomState(2)
        values = refined_composite_multiscale_sample_entropy(
            rng.randn(400), m=2, r=0.15, scale=3
        )
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.any(np.isfinite(values)))


class ApproximateEntropyTest(unittest.TestCase):
    """Pincus approximate entropy."""

    def test_finite_sine(self) -> None:
        value = approximate_entropy(_sine(), m=2, r=0.2)
        self.assertTrue(np.isfinite(value))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            approximate_entropy(np.array([1.0]), m=1, r=0.2)

    def test_embedding_too_large(self) -> None:
        with self.assertRaises(ValueError):
            approximate_entropy(_sine(10), m=10, r=0.2)


class FuzzyEntropyTest(unittest.TestCase):
    """Fuzzy entropy and multiscale fuzzy entropy."""

    def test_finite_sine(self) -> None:
        value = fuzzy_entropy(_sine(), m=2, r=0.2)
        self.assertTrue(np.isfinite(value))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            fuzzy_entropy(np.arange(3.0), m=2, r=0.2)

    def test_multiscale_length(self) -> None:
        values = multiscale_fuzzy_entropy(_sine(), m=2, r=0.2, scale=3)
        self.assertEqual(len(values), 3)


class DispersionEntropyTest(unittest.TestCase):
    """Dispersion entropy and its multiscale variant."""

    def test_finite_sine(self) -> None:
        value = dispersion_entropy(_sine(), m=2, c=3)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_invalid_classes(self) -> None:
        with self.assertRaises(ValueError):
            dispersion_entropy(_sine(), m=2, c=1)

    def test_multiscale_length(self) -> None:
        values = multiscale_dispersion_entropy(_sine(), m=2, c=3, scale=4)
        self.assertEqual(len(values), 4)
        self.assertTrue(np.all(np.isfinite(values)))


class SpectralEntropyTest(unittest.TestCase):
    """Periodogram Shannon entropy."""

    def test_finite_sine(self) -> None:
        value = spectral_entropy(_sine())
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0 + 1e-9)

    def test_noise_higher_than_sine(self) -> None:
        sine_h = spectral_entropy(_sine(512))
        noise_h = spectral_entropy(_noise(512, seed=3), normalize=True)
        self.assertGreater(noise_h, sine_h)

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            spectral_entropy(np.array([1.0]))


class DistributionEntropyTest(unittest.TestCase):
    """Distribution entropy (distance histogram)."""

    def test_finite_sine(self) -> None:
        value = distribution_entropy(_sine(), m=2)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_embedding_too_large(self) -> None:
        with self.assertRaises(ValueError):
            distribution_entropy(_sine(8), m=8)


class IncrementEntropyTest(unittest.TestCase):
    """Increment entropy."""

    def test_finite_sine(self) -> None:
        value = increment_entropy(_sine(), m=2, r=4)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            increment_entropy(np.array([1.0, 2.0]), m=2)


class SlopeEntropyTest(unittest.TestCase):
    """Slope entropy."""

    def test_finite_sine(self) -> None:
        value = slope_entropy(_sine(), m=3)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_m_too_small(self) -> None:
        with self.assertRaises(ValueError):
            slope_entropy(_sine(), m=1)

    def test_bad_levels(self) -> None:
        with self.assertRaises(ValueError):
            slope_entropy(_sine(), m=2, levels=(50.0, 10.0))


class SymbolicDynamicEntropyTest(unittest.TestCase):
    """Equal-width symbolic dynamic entropy."""

    def test_finite_sine(self) -> None:
        value = symbolic_dynamic_entropy(_sine(), m=2, c=4)
        self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(value, 0.0)

    def test_invalid_classes(self) -> None:
        with self.assertRaises(ValueError):
            symbolic_dynamic_entropy(_sine(), m=2, c=1)

    def test_embedding_too_large(self) -> None:
        with self.assertRaises(ValueError):
            symbolic_dynamic_entropy(_sine(10), m=10, c=3)


if __name__ == "__main__":
    unittest.main()
