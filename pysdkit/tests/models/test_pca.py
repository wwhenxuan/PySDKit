# -*- coding: utf-8 -*-
"""
Created on 2025/07/22 15:21:32
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit.models import PCA
from pysdkit.data import test_pca


class PCATest(unittest.TestCase):
    """Test whether PCA runs normally."""

    def test_create_pca(self) -> None:
        """Verify that PCA can be created correctly."""
        for n_components in range(2, 6):
            pca = PCA(n_components=n_components)
            self.assertIsInstance(pca, PCA, msg="Created PCA object has the wrong type")
            self.assertEqual(
                pca.n_components,
                n_components,
                msg="PCA attribute does not match the input hyperparameter",
            )

    def test_none(self) -> None:
        """Initialized private attributes should be None before fitting."""
        pca = PCA(n_components=2)
        self.assertIsNone(
            pca._X_reduced, msg="Created PCA object has non-None data before fitting"
        )
        self.assertIsNone(
            pca._components, msg="Created PCA object has non-None data before fitting"
        )
        self.assertIsNone(
            pca._explained_variance_ratio,
            msg="Created PCA object has non-None data before fitting",
        )

    def test_wrong_shape_inputs(self) -> None:
        """Invalid input shapes should raise ValueError."""
        pca = PCA(n_components=2)
        with self.assertRaises(ValueError):
            pca.fit_transform(X=np.ones(100))
        with self.assertRaises(ValueError):
            pca.fit_transform(X=np.ones(shape=(10, 10, 100)))

    def test_wrong_type_inputs(self) -> None:
        """Invalid input types should raise TypeError."""
        pca = PCA(n_components=2)
        with self.assertRaises(TypeError):
            pca.fit_transform(X=42)

    def test_fit_transform(self) -> None:
        """fit_transform should reduce dimensionality to n_components."""
        X = test_pca(number=200, dim=4)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        self.assertEqual(X_reduced.shape, (X.shape[0], 2))
        self.assertEqual(np.asarray(pca.components_).shape[0], 2)
        self.assertEqual(len(pca.explained_variance_ratio_), 2)
        self.assertTrue(
            np.isclose(np.sum(pca.explained_variance_ratio_), 1.0, atol=1e-6)
            or np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-6
        )

    def test_properties_before_fit(self) -> None:
        """Accessing fitted properties before fit should raise ValueError."""
        pca = PCA(n_components=2)
        with self.assertRaises(ValueError):
            _ = pca.components_
        with self.assertRaises(ValueError):
            _ = pca.explained_variance_ratio_


if __name__ == "__main__":
    unittest.main()
