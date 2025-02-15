# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 23:21:53
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import Moving_Decomp
from pysdkit.data import generate_time_series


class MDTest(unittest.TestCase):
    """Moving_Decomp test"""

    def test_univariate_fit_transform(self):
        """验证能否进行一元信号或时间序列的分解"""
        # 创建分解的实例化对象
        moving_decomp = Moving_Decomp(window_size=5)

        # univariate time series
        time_series = generate_time_series()
        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # 验证一元时间序列的输出
        self.assertEqual(
            first=len(time_series), second=len(trends), msg="输入输出信号的长度不一致"
        )
        self.assertEqual(
            first=len(time_series),
            second=len(seasonalities),
            msg="输入输出信号的长度不一致",
        )

        # 进一步判断分解的数值差异
        diff = np.allclose(time_series, trends + seasonalities)
        self.assertTrue(expr=diff, msg="分解得到的趋势和周期无法重构原信号")

    def test_multivariate_fit_transform(self):
        """验证能否进行多元信号或时间序列的分解"""
        # 创建分解的实例化对象
        moving_decomp = Moving_Decomp(window_size=5)

        # multivariate time series
        time_series = np.vstack(
            [
                generate_time_series(periodicities=np.array(p_list))
                for p_list in [[10, 20, 30], [5, 20, 50], [40, 60, 10]]
            ]
        )

        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # 验证输入输出的形状是否匹配
        self.assertEqual(
            first=trends.shape,
            second=time_series.shape,
            msg="分解趋势的形状应该与输入序列相同",
        )
        self.assertEqual(
            first=seasonalities.shape,
            second=time_series.shape,
            msg="分解周期的形状应该与输入序列相同",
        )

        # 进一步判断分解的数值差异
        for index in range(3):
            diff = np.allclose(time_series[index], trends[index] + seasonalities[index])
            self.assertTrue(expr=diff, msg="分解得到的趋势和周期无法重构原信号")

    def test_default_call(self) -> None:
        """验证call方法嫩否正常运行"""
        # 创建分解的实例化对象
        moving_decomp = Moving_Decomp(window_size=5)

        # univariate time series
        time_series = generate_time_series()
        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # 验证一元时间序列的输出
        self.assertEqual(
            first=len(time_series), second=len(trends), msg="输入输出信号的长度不一致"
        )
        self.assertEqual(
            first=len(time_series),
            second=len(seasonalities),
            msg="输入输出信号的长度不一致",
        )

        # 进一步判断分解的数值差异
        diff = np.allclose(time_series, trends + seasonalities)
        self.assertTrue(expr=diff, msg="分解得到的趋势和周期无法重构原信号")

    def test_different_window_size(self) -> None:
        """验证不同滑动平均窗口的大小对算法的影响"""
        # univariate time series
        time_series = generate_time_series()

        # 存放不同滑动窗口大小的列表
        window_size_list = [3, 5, 7, 9, 15, 25, 35]

        for window_size in window_size_list:
            # 根据不同的滑动窗口大小创建分解实例
            moving_decomp = Moving_Decomp(window_size=window_size)
            trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

            # 验证一元时间序列的输出
            self.assertEqual(
                first=len(time_series),
                second=len(trends),
                msg="输入输出信号的长度不一致",
            )
            self.assertEqual(
                first=len(time_series),
                second=len(seasonalities),
                msg="输入输出信号的长度不一致",
            )

            # 进一步判断分解的数值差异
            diff = np.allclose(time_series, trends + seasonalities)
            self.assertTrue(expr=diff, msg="分解得到的趋势和周期无法重构原信号")

    def test_different_method(self) -> None:
        """验证不同滑动平均分解的方法"""
        # univariate time series
        time_series = generate_time_series()

        # 存放不同方法的列表
        methods = ["simple", "weighted", "gaussian", "savgol", "exponential"]

        for method in methods:
            # 根据不同的滑动平均方法创建分解实例
            moving_decomp = Moving_Decomp(method=method)
            trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

            # 验证一元时间序列的输出
            self.assertEqual(
                first=len(time_series),
                second=len(trends),
                msg="输入输出信号的长度不一致",
            )
            self.assertEqual(
                first=len(time_series),
                second=len(seasonalities),
                msg="输入输出信号的长度不一致",
            )

            # 进一步判断分解的数值差异
            diff = np.allclose(time_series, trends + seasonalities)
            self.assertTrue(expr=diff, msg="分解得到的趋势和周期无法重构原信号")

    def test_methods_list(self) -> None:
        """验证再fit_transform中传递分解方法的列表"""
        moving_decomp = Moving_Decomp()

        # univariate time series
        time_series = generate_time_series()
        # multivariate time series
        time_series = np.vstack([time_series] * 3)

        trends, seasonalities = moving_decomp.fit_transform(signal=time_series, methods_list=["simple", "weighted", "gaussian"])



if __name__ == "__main__":
    unittest.main()
