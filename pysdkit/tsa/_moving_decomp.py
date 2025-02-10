# -*- coding: utf-8 -*-
"""
Created on 2025/02/10 13:14:41
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt

from pysdkit.utils import (
    simple_moving_average,
    weighted_moving_average,
    gaussian_smoothing,
    savgol_smoothing,
    exponential_smoothing,
)

from typing import Optional, Tuple, List


class Moving_Decomp(object):
    """
    通过滑动平均的方式进行一维信号的分解，将其分解为趋势和周期两部分内容。
    该方法十分简单，非常适合用于处理非平稳的时间序列数据。
    在
    """

    def __init__(
        self,
        window_size: int = 5,
        method: str = "simple",
        sigma: int = 2,
        poly_order: int = 2,
        alpha: float = 0.4,
    ) -> None:
        """
        通过滑动平均的方式对输入的信号进行分解，得到趋势和周期两部分内容
        :param window_size:
        :param method:
        :param sigma:
        :param poly_order:
        :param alpha:
        """
        self.window_size = window_size
        self.method = method

        self.sigma = sigma
        self.poly_order = poly_order
        self.alpha = alpha

        # 存放所有方法的列表
        self.method_list = ["simple", "weighted", "gaussian", "savgol", "exponential"]
        if self.method not in self.method_list:
            # 使用的平滑方法错误
            raise ValueError("method must be one of {}".format(self.method_list))

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        pass

    def _decomposition(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行一次滑动平均分解算法
        要求输入的信号必须是一元信号

        :param signal: the input univariate signal of 1D numpy ndarray
        :return: the trend and seasonality of the input signal
        """
        # 使用具体的滑动平均分解方法
        if self.method == "simple":
            trend = simple_moving_average(signal=signal, window_size=self.window_size)
        elif self.method == "weighted":
            trend = weighted_moving_average(signal=signal, window_size=self.window_size)
        elif self.method == "gaussian":
            trend = gaussian_smoothing(signal=signal, sigma=self.sigma)
        elif self.method == "savgol":
            trend = savgol_smoothing(
                signal=signal,
                window_length=self.window_size,
                poly_order=self.poly_order,
            )
        elif self.method == "exponential":
            trend = exponential_smoothing(signal=signal, alpha=self.alpha)
        else:
            raise ValueError(
                "method must be 'simple' or 'weighted' or 'gaussian' or 'savgol' or 'exponential'"
            )

        # 从原始的输入信号中减去趋势部分
        seasonality = signal - trend

        # 同时返回趋势和季节性分量
        return trend, seasonality

    def fit_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        # TODO: 可以指定传入一个列表表示对于每一个通道的时间序列到底采用哪种分解方法
        # 检验输入信号的维度
        shape = signal.shape

        if len(shape) == 1:
            # 输入为一维的一元信号
            trend, seasonality = self._decomposition(signal=signal)

        elif len(shape) == 2:
            # 输入为一维的多元信号
            # 获取输入信号的数目
            n_vars, seq_len = shape

            # 初始化分解的数组
            trend, seasonality = np.zeros(shape=(n_vars, seq_len)), np.zeros(
                shape=(n_vars, seq_len)
            )

            for n in range(n_vars):
                # 遍历每一个信号进行滑动平均分解
                trend[n, :], seasonality[n, :] = self._decomposition(
                    signal=signal[n, :]
                )
        else:
            raise ValueError(
                "The input must be 1D univariate or multivariate signal with shape [seq_len] or [n_vars, seq_len]"
            )

        # 返回分解后的结果
        return trend, seasonality

    @staticmethod
    def plot_decomposition(
        signal: np.ndarray,
        trend: np.ndarray,
        seasonality: np.ndarray,
        colors: List[str] = None,
    ) -> Optional[plt.Figure]:
        """
        对输入信号分解后的结果进行可视化
        :param signal:
        :param trend:
        :param seasonality:
        :param colors:
        :return:
        """
        # 通过输入数据的形状判断其维数
        if colors is None:
            colors = ["royalblue", "royalblue", "royalblue"]
        shape = signal.shape

        # If the inputs is univariate signal
        if len(shape) == 1:
            # 创建绘图对象
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=True)
            # 开始绘制图像
            ax[0].plot(signal, color=colors[0])
            ax[1].plot(trend, color=colors[1])
            ax[2].plot(seasonality, color=colors[2])

            ax[0].set_ylabel("input signal")
            ax[1].set_ylabel("trend")
            ax[2].set_ylabel("seasonality")

        elif len(shape) == 2:
            # 获取输入信号的数目
            n_vars, seq_len = shape
            # 创建绘图对象
            fig, ax = plt.subplots(
                nrows=3, ncols=n_vars, figsize=(4 * n_vars, 5), sharex=True
            )
            # 遍历所有维度的变量绘制图像
            for n in range(n_vars):
                ax[0, n].plot(signal[n, :], color=colors[n])
                ax[1, n].plot(trend[n, :], color=colors[n])
                ax[2, n].plot(seasonality[n, :], color=colors[n])

                ax[0, n].set_title(f"Input channels {n}")

            ax[0, 0].set_ylabel("input signal")
            ax[1, 0].set_ylabel("trend")
            ax[2, 0].set_ylabel("seasonality")

        else:
            raise ValueError(
                "The input must be 1D univariate or multivariate signal with shape [seq_len] or [n_vars, seq_len]"
            )

        return fig


if __name__ == "__main__":
    from pysdkit.data import generate_time_series

    # univariate time series
    time_series = generate_time_series()

    moving_decomp = Moving_Decomp(window_size=5)

    trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

    moving_decomp.plot_decomposition(
        signal=time_series, trend=trends, seasonality=seasonalities
    )
    plt.show()

    # multivariate time series
    time_series = np.vstack([time_series] * 3)

    trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

    moving_decomp.plot_decomposition(
        signal=time_series, trend=trends, seasonality=seasonalities
    )
    plt.show()
