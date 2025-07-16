# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 15:54:22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import LMD
from pysdkit.data import test_emd, test_univariate_signal


class LMDTest(unittest.TestCase):
    """测试经验模态分解算法(EMD)能否正常运行"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""
        # 创建算法实例对象
        lmd = LMD()
        for index in range(1, 4):
            # 获取测试信号
            time, signal = test_univariate_signal(case=index)
            IMFs = lmd.fit_transform(signal)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            _, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")

    def test_default_call(self) -> None:
        """验证call方法能否正常运行"""
        time, signal = test_emd()
        # 创建算法实例对象
        lmd = LMD()
        IMFs = lmd(signal)

        # 判断输出的维数
        dim = len(IMFs.shape)
        self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
        # 判断输出信号的长度
        _, length = IMFs.shape
        self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")




if __name__ == "__main__":
    unittest.main()
