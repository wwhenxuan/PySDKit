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
        # 生成测试函数样例
        time, signal = test_emd()
        # 遍历不同的分解模态数目
        for K in range(1, 4):
            # 创建信号分解实例对象
            ewt = EWT(K=K)
            # 执行信号分解算法
            IMFs = ewt.fit_transform(signal)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            number, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")
            # 判断分解模态的数目
            self.assertEqual(first=number, second=K)

    def test_default_call(self) -> None:
        """验证call方法能够正常运行"""
        # 生成测试函数样例
        time, signal = test_emd()
        # 遍历不同的分解模态数目
        for K in range(1, 4):
            # 创建信号分解实例对象
            ewt = EWT(K=K)
            # 执行信号分解算法
            IMFs = ewt(signal)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            number, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")
            # 判断分解模态的数目
            self.assertEqual(first=number, second=K)

    def test_ewt_function(self) -> None:
        """测试经验小波变换的函数"""

    def test_trend(self) -> None:
        """验证算法能否分解出趋势信息"""

    def test_signal_imf(self) -> None:
        """验证单一本征模态函数的输入"""

    def test_fmirror(self) -> None:
        """验证EWT算法自带的镜像拓展算法"""

    def test_detect(self) -> None:
        """测试边界检测的各种方法"""

    def test_reg(self) -> None:
        """Test the regularization method applied to the filter bank"""




if __name__ == '__main__':
    unittest.main()