# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 18:26:17
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import VMD
from pysdkit import vmd as VMD_f
from pysdkit.data import test_emd


class Test(unittest.TestCase):
    """对变分模态分解(VMD)算法进行自动化测试"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""
        # 创建算法实例对象
        vmd = VMD(K=3, alpha=1000, tau=0.0)

        time, signal = test_emd()
        IMFs = vmd.fit_transform(signal)
        # 判断输出的维数
        dim = len(IMFs.shape)
        self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
        # 判断输出信号的长度
        _, length = IMFs.shape
        self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")

    def test_default_call(self) -> None:
        """验证call方法嫩否正常运行"""
        time, signal = test_emd()
        # 创建算法实例对象
        vmd = VMD(K=3, alpha=1000, tau=0.0)
        IMFs = vmd(signal)

        # 判断输出的维数
        dim = len(IMFs.shape)
        self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
        # 判断输出信号的长度
        _, length = IMFs.shape
        self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")

    def test_vmd_function(self) -> None:
        """判断VMD函数接口能否正常运行"""
        time, signal = test_emd()
        # 创建算法实例对象
        IMFs, _, _ = VMD_f(signal, alpha=1000, K=3, tau=0.0)

        # 判断输出的维数
        dim = len(IMFs.shape)
        self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
        # 判断输出信号的长度
        _, length = IMFs.shape
        self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")

    def test_trend(self) -> None:
        """判断对单一的趋势信号输入"""
        vmd = VMD(K=1, alpha=1000, tau=0.0)

        # 创建仅有趋势分类的时间戳和信号
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # 执行信号分解算法获得本征模态函数
        IMFs = vmd.fit_transform(signal=signal)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF")
        self.assertTrue(np.allclose(signal, IMFs[0], atol=0.1))

    def test_single_imf(self) -> None:
        """判断单一本征模态函数的输入"""
        vmd = VMD(K=1, alpha=1000, tau=0.0)

        # 创建时间戳数组
        time = np.arange(0, 1, 0.001)

        # 创建余弦信号
        cosine = np.cos(2 * np.pi * 4 * time)

        # 判断单一余弦函数的输入
        IMFs = vmd.fit_transform(signal=cosine.copy())
        # 判断输入输出之间的数值差异
        diff = np.allclose(IMFs[0], cosine, atol=0.1)
        self.assertTrue(diff, "Expecting 1st IMF to be cos(8 * pi * t)")

        # 创建输入的趋势分量
        trend = 3 * (time - 0.5)
        vmd = VMD(K=2, alpha=1000, tau=0.0)

        # 判断余弦与趋势分量输入
        IMFs = vmd.fit_transform(signal=trend.copy() + cosine.copy())
        self.assertEqual(
            first=IMFs.shape[0], second=2, msg="Expecting two IMF of cosine and trend!"
        )

        # 进一步判断两个模态输出的数值差异
        diff_cosine = np.allclose(IMFs[0], trend, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 1st IMF to be trend")

        diff_trend = np.allclose(IMFs[1], cosine, atol=0.2)
        self.assertTrue(diff_trend, "Expecting 2nd IMF to be cosine")

    def test_vmd_DC(self) -> None:
        """判断VMD能否分离出主流信号分量"""
        # 创建VMD算法实例
        vmd = VMD(K=2, alpha=1000, tau=0.0, DC=True)

        # 创建带有直流分量的信号
        time = np.arange(0, 1, 0.001)
        DC = 2
        cosine = np.cos(2 * np.pi * 4 * time)
        signal = DC + cosine.copy()

        # 开始执行信号分解算法
        IMFs = vmd.fit_transform(signal=signal)

        # 首先判断信号的数目
        self.assertEqual(first=IMFs.shape[0], second=2, msg="Expecting the number of IMFs is Two")

        # 进一步判断两个模态输出的数值差异
        diff_DC = np.allclose(IMFs[0], np.ones_like(time) * DC, atol=0.1)
        self.assertTrue(diff_DC, "Expecting 1st IMF to be DC")

        diff_cosine = np.allclose(IMFs[1], cosine, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 2nd IMF to be cosine")



if __name__ == "__main__":
    unittest.main()
