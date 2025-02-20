# -*- coding: utf-8 -*-
"""
Created on 2025/02/20 22:44:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import VMD2D
from pysdkit.data import test_univariate_image
from pysdkit.data import test_grayscale


class VMD2DTest(unittest.TestCase):
    """验证二维变分模态分解算法"""
    image = test_univariate_image(case=1)
    grayscale = test_grayscale()

    def test_fit_transform(self) -> None:
        """验证算法能够正常进行信号分解"""
        # 分解模态的数目
        K = 5

        # 创建二维信号分解的测试实例
        vmd2d = VMD2D(K=K, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=64)

        # 执行信号分解算法
        IMFs = vmd2d.fit_transform(self.grayscale)

        # 判断生成图像的长宽是否与输入图像相匹配
        H, W, C = IMFs.shape
        self.assertEqual(first=H, second=self.grayscale.shape[0], msg="分解图像的高度尺寸错误")
        self.assertEqual(first=W, second=self.grayscale.shape[1], msg="分解图像的宽度尺寸错误")
        self.assertEqual(first=C, second=K, msg="算法分解的模态数目错误")
