# -*- coding: utf-8 -*-
"""
Created on 2025/07/22 10:08:57
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit.models import KNN


class KNNTest(unittest.TestCase):
    """Test whether the K-Nearest Neighbors classifier runs normally"""

    # Create a random number generator for testing
    rng = np.random.RandomState(42)

    def test_create_knn(self) -> None:
        """Test whether the KNN classifier can be created correctly"""
        for n_neighbors in range(1, 6):
            # Create an instance of the class
            knn = KNN(n_neighbors=n_neighbors)
            # Verify that the created instance has the expected type
            self.assertIsInstance(knn, KNN, msg="Created KNN object has the wrong type")
            # Verify that class attributes match the input hyperparameter
            self.assertEqual(
                knn.n_neighbors,
                n_neighbors,
                msg="KNN attribute does not match the input hyperparameter",
            )

    def test_none(self) -> None:
        """Test whether initialized data attributes are None before calling `fit`"""
        # Create a KNN classifier instance
        knn = KNN(n_neighbors=1)
        # Verify that fitted data is initially None
        self.assertIsNone(
            knn.X, msg="Created KNN object has non-None data before fitting"
        )
        self.assertIsNone(
            knn.y, msg="Created KNN object has non-None data before fitting"
        )

    def test_wrong_shape_inputs(self) -> None:
        """Test invalid shape inputs for the KNN classifier"""
        # Create an algorithm instance
        knn = KNN(n_neighbors=1)

        # Test invalid sample input with one missing dimension
        with self.assertRaises(ValueError):
            knn.fit(X=np.ones(100), y=np.ones(100))

        # Test invalid sample input with one extra dimension
        with self.assertRaises(ValueError):
            knn.fit(X=np.ones(shape=(100, 100, 100)), y=np.ones(100))

        # Test invalid label input with one extra dimension
        with self.assertRaises(ValueError):
            knn.fit(X=np.ones(shape=(100, 100)), y=np.ones(shape=(100, 100)))

        # Test mismatched sample and label counts
        with self.assertRaises(ValueError):
            knn.fit(X=np.ones(shape=(100, 50)), y=np.ones(99))

    def test_wrong_type_inputs(self) -> None:
        """Test invalid data type inputs for the KNN classifier"""
        # Create an algorithm instance
        knn = KNN(n_neighbors=1)

        # Test invalid sample input type
        with self.assertRaises(TypeError):
            knn.fit(42, np.ones(100))

        # Test invalid label input type
        with self.assertRaises(TypeError):
            knn.fit(np.ones(shape=(10, 100)), 1)

    def test_fit(self) -> None:
        """Test whether the `fit` method executes correctly"""
        # Generate random data for testing
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # Create a KNN classifier instance
        knn = KNN(n_neighbors=1)

        # Fit the data
        knn.fit(X, y)

        # Verify fitted data
        self.assertEqual(
            first=X.all(),
            second=knn.X.all(),
            msg="Input samples do not match fitted samples",
        )
        self.assertEqual(
            first=y.all(),
            second=knn.y.all(),
            msg="Input labels do not match fitted labels",
        )

    def test_predict(self) -> None:
        """Test the prediction method in the KNN classifier"""
        # Generate random data for testing
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # Create a KNN classifier instance
        knn = KNN(n_neighbors=1)

        # Fit the data
        knn.fit(X, y)
        # Predict labels
        (y_pred, y_prob) = knn.predict(X)

        # Verify that labels match before and after prediction
        self.assertEqual(
            first=y.all(), second=y_pred.all(), msg="KNN classification is incorrect"
        )

        # Verify label and probability lengths
        self.assertEqual(
            first=len(y), second=len(y_pred), msg="Wrong output label length"
        )
        self.assertEqual(
            first=len(y), second=len(y_prob), msg="Wrong output probability length"
        )

    def test_fit_transform(self) -> None:
        """Test whether the KNN `fit_transform` method executes correctly"""
        # Generate random data for testing
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # Create a KNN classifier instance
        knn = KNN(n_neighbors=1)

        # Fit the data and predict results
        (y_pred, y_prob) = knn.fit_transform(X, y, X)

        # Verify that labels match before and after prediction
        self.assertEqual(
            first=y.all(), second=y_pred.all(), msg="KNN classification is incorrect"
        )

        # Verify label and probability lengths
        self.assertEqual(
            first=len(y), second=len(y_pred), msg="Wrong output label length"
        )
        self.assertEqual(
            first=len(y), second=len(y_prob), msg="Wrong output probability length"
        )

    def test_call(self) -> None:
        """Test whether the call method executes correctly"""
        # Generate random data for testing
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # Create a KNN classifier instance
        knn = KNN(n_neighbors=1)

        # Fit the data and predict results
        (y_pred, y_prob) = knn(X, y, X)

        # Verify that labels match before and after prediction
        self.assertEqual(
            first=y.all(), second=y_pred.all(), msg="KNN classification is incorrect"
        )

        # Verify label and probability lengths
        self.assertEqual(
            first=len(y), second=len(y_pred), msg="Wrong output label length"
        )
        self.assertEqual(
            first=len(y), second=len(y_prob), msg="Wrong output probability length"
        )

    def test_str(self) -> None:
        """Verify that `__str__` returns a string correctly"""
        # Create a KNN algorithm instance
        knn = KNN(n_neighbors=1)

        # Verify that a string object is returned
        self.assertIsInstance(str(knn), str, "`__str__` did not return a string")


if __name__ == "__main__":
    unittest.main()
