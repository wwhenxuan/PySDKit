# -*- coding: utf-8 -*-
"""
Created on 2025/02/21 23:40:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import EWT
from pysdkit.data import test_emd


class EWTTest(unittest.TestCase):
    """测试经验小波变换算法(EMD)能否正常运行"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""

    def test_default_call(self) -> None:
        """验证call方法能够正常运行"""



if __name__ == '__main__':
    unittest.main()