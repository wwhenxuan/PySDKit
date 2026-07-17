# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 15:54:38
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

from pysdkit import RLMD
from pysdkit.data import test_emd, test_univariate_signal


class RLMDTest(unittest.TestCase):
    """Test whether Robust Local Mean Decomposition (RLMD) runs normally"""

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Create an algorithm instance object
        rlmd = RLMD()
        for index in range(1, 4):
            # Get the test signal
            time, signal = test_univariate_signal(case=index)
            IMFs = rlmd.fit_transform(signal)
            # Determine the output dimension
            dim = len(IMFs.shape)
            self.assertEqual(
                first=dim,
                second=2,
                msg="The output shape of the decomposed signal is wrong",
            )
            # Determine the length of the output signal
            _, length = IMFs.shape
            self.assertEqual(
                first=len(signal),
                second=length,
                msg="Wrong length of decomposed signal",
            )

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        time, signal = test_emd()
        # Create an algorithm instance object
        rlmd = RLMD()
        IMFs = rlmd(signal)

        # Determine the output dimension
        dim = len(IMFs.shape)
        self.assertEqual(
            first=dim,
            second=2,
            msg="The output shape of the decomposed signal is wrong",
        )
        # Determine the length of the output signal
        _, length = IMFs.shape
        self.assertEqual(
            first=len(signal), second=length, msg="Wrong length of decomposed signal"
        )

    def test_imfs_number(self) -> None:
        """Verify that the number of IMFs matches the specified hyperparameter"""
        time, signal = test_emd()

        # Iterate over multiple hyperparameter values
        for k in [2, 3, 5]:
            # Create an algorithm instance and run the algorithm
            rlmd = RLMD(max_imfs=k)
            IMFs = rlmd.fit_transform(signal)

            # Verify that the number of decomposed modes matches the hyperparameter
            number = IMFs.shape[0]
            # Account for the residue component, so expect k + 1
            self.assertEqual(
                first=number,
                second=k + 1,
                msg="The number of IMFs does not match the specified hyperparameter",
            )


if __name__ == "__main__":
    unittest.main()
