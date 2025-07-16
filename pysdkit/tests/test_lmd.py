# -*- coding: utf-8 -*-
"""
Created on 2025/07/16 15:54:22
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

from pysdkit import LMD
from pysdkit.data import test_emd, test_univariate_signal


class LMDTest(unittest.TestCase):
    """测试局部均值分解(LMD)算法能否正常运行"""

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

    def test_imfs_number(self) -> None:
        """验证算法分解得到的本征模态函数的数目和指定的超参数是否一致"""
        time, signal = test_emd()

        # 遍历多个超参数
        for k in [2, 3, 5]:
            # 创建算法实例并执行算法
            lmd = LMD(K=k)
            IMFs = lmd.fit_transform(signal)

            # 判断分解得到的模态数目和指定的超参数数目是否一致
            number = IMFs.shape[0]
            # 考虑到有残差的存在因此应该要+1
            self.assertEqual(first=number, second=k + 1, msg="分解得到的本征模态函数的数目和指定的超参数不一致")



if __name__ == "__main__":
    unittest.main()
