# -*- coding: utf-8 -*-
"""
Created on 2025/02/20 22:44:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import VMD2D
from pysdkit.data import test_univariate_image, test_grayscale, get_meshgrid_2D


class VMD2DTest(unittest.TestCase):
    """验证二维变分模态分解算法"""

    image = test_univariate_image(case=1)
    grayscale = test_grayscale()

    def test_fit_transform(self) -> None:
        """验证算法能够正常进行信号分解"""
        # 分解模态的数目
        K = 5

        # 创建二维信号分解的测试实例
        vmd2d = VMD2D(
            K=K, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=64
        )

        # 执行信号分解算法
        IMFs = vmd2d.fit_transform(self.grayscale)

        # 判断生成图像的长宽是否与输入图像相匹配
        H, W, C = IMFs.shape
        self.assertEqual(
            first=H, second=self.grayscale.shape[0], msg="分解图像的高度尺寸错误"
        )
        self.assertEqual(
            first=W, second=self.grayscale.shape[1], msg="分解图像的宽度尺寸错误"
        )
        self.assertEqual(first=C, second=K, msg="算法分解的模态数目错误")

    def test_default_call(self) -> None:
        """验证call方法嫩否正常运行"""
        # 分解模态的数目
        K = 5

        # 创建二维信号分解的测试实例
        vmd2d = VMD2D(
            K=K, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=64
        )

        # 执行信号分解算法
        IMFs = vmd2d(self.grayscale)

        # 判断生成图像的长宽是否与输入图像相匹配
        H, W, C = IMFs.shape
        self.assertEqual(
            first=H, second=self.grayscale.shape[0], msg="分解图像的高度尺寸错误"
        )
        self.assertEqual(
            first=W, second=self.grayscale.shape[1], msg="分解图像的宽度尺寸错误"
        )
        self.assertEqual(first=C, second=K, msg="算法分解的模态数目错误")

    def test_VMD2D_DC(self) -> None:
        """判断VMD2D能否分离出直流信号分量"""

        # 创建信号分解算法
        vmd2d = VMD2D(
            K=2, alpha=5000, tau=0.0, DC=True, init="random", tol=1e-6, max_iter=5000
        )

        for size in [16, 32, 64]:
            # 生成简单的直流分量
            DC = np.ones(shape=(size, size)) * 100

            # 生成网格矩阵
            x, y = get_meshgrid_2D(low=0, high=2 * np.pi, sampling_rate=size)

            # 生成一个模态的函数
            mode = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)

            # 合成2D信号
            image = DC + mode

            # 执行信号分解算法
            IMFs = vmd2d.fit_transform(image)

            # 判断能否提取出直流分量
            diff_DC = np.allclose(IMFs[:, :, 0], DC, atol=1)
            self.assertTrue(expr=diff_DC, msg="VMD2D未能成功提取直流分量")

    def test_init_omega(self) -> None:
        """验证VMD2D算法对omega_k的多种初始化方式"""
        # 遍历几种不同的初始化方式
        for init in ["random", "zero"]:
            # 根据输入的参数创建算法实例
            vmd2d = VMD2D(K=5, alpha=5000, tau=0.25, init=init, tol=1e-6, max_iter=64)
            # 执行信号分解算法
            IMFs = vmd2d.fit_transform(self.grayscale)
            self.assertEqual(first=IMFs.shape[-1], second=5, msg="VMD2D算法输出错误")

    def test_wrong_init_omega(self) -> None:
        """验证错误的omega_k初始化方式"""
        init = "uniform"
        # 使用错误的参数创建算法实例
        with self.assertRaises(ValueError):
            vmd2d = VMD2D(K=5, alpha=5000, tau=0.25, init=init, tol=1e-6, max_iter=64)
            vmd2d.fit_transform(self.grayscale)

    def test_return_all(self) -> None:
        """判断VMD2D算法是否能通过参数控制返回所有的信息"""

        # 创建VMD算法并获得全部的输出
        vmd = VMD2D(
            K=5, alpha=5000, tau=0.25, DC=False, init="random", tol=1e-6, max_iter=64
        )
        outputs = vmd.fit_transform(self.grayscale, return_all=True)

        # 判断模型返回变量的数目
        self.assertEqual(first=len(outputs), second=3, msg="Expecting three outputs")

        # 解耦全部的变量
        u, u_hat, omega = outputs

        # 判断频率分量信息
        self.assertEqual(
            first=len(u_hat.shape), second=3, msg="Expecting three dimensions of u_hat"
        )
        self.assertEqual(
            first=len(omega.shape), second=3, msg="Expecting three dimensions of omega"
        )


if __name__ == "__main__":
    unittest.main()
