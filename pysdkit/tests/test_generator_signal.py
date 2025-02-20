# -*- coding: utf-8 -*-
"""
Created on 2025/02/16 00:11:14
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit.data import add_noise, generate_cos_signal, generate_sin_signal


class SignalTest(unittest.TestCase):
    """验证各种生成一维信号的算法"""

    def test_add_noise(self) -> None:
        """验证添加噪声的函数"""
        for N in [10, 100, 500, 1000]:
            # 遍历不同的信号长度
            for mean in [0.0, 0.1, -0.1, 0.5, -0.5, 5, 10, 20, 100, -100]:
                # 遍历不同的均值
                for std in [0.1, 0.5, 1, 2, 3, 5]:
                    # 遍历不同的标准差
                    y = add_noise(N, mean, std)
                    self.assertTrue(expr=len(y) == N, msg="输出长度错误")
                    # 验证噪声均值的数值差异
                    diff_mean = np.allclose(np.mean(y), mean, atol=1e-6)
                    self.assertTrue(expr=diff_mean, msg="输出噪声的均值不符合要求")
                    # 验证噪声标准差的数值差异
                    diff_std = np.allclose(np.std(y), std, atol=1e-6)
                    self.assertTrue(expr=diff_std, msg="输出噪声的标准差不符合要求")


if __name__ == '__main__':
    unittest.main()
