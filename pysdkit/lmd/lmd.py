# -*- coding: utf-8 -*-
"""
Created on 2025/01/31 21:35:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.signal import argrelextrema

from typing import Optional, Tuple


class LMD(object):
    """
    Local Mean Decomposition
    Jia, Linshan, et al. “The Empirical Optimal Envelope and Its Application to Local Mean Decomposition.”
    Digital Signal Processing, vol. 87, Elsevier BV, Apr. 2019, pp. 166–77, doi:10.1016/j.dsp.2019.01.024.
    Python Link: https://github.com/shownlin/PyLMD
    MATLAB Link: https://www.mathworks.com/matlabcentral/fileexchange/107829-local-mean-decomposition?s_tid=srchtitle
    """

    def __init__(self, K: int = 5, endpoints: bool = True, max_smooth_iter: int = 15, max_envelope_iter: int = 200,
                 envelope_epsilon: float = 0.01, convergence_epsilon: float = 0.01,
                 min_extrema: int = 5) -> None:
        """
        :param K: the maximum number of IFMs to be decomposed
        :param endpoints: whether to treat the endpoint of the signal as a pseudo-extreme point
        :param max_smooth_iter: maximum number of iterations of moving average algorithm
        :param max_envelope_iter: maximum number of iterations when separating local envelope signals
        :param envelope_epsilon: terminate processing when obtaining pure FM signal
        :param convergence_epsilon: terminate processing when modulation signal converges
        :param min_extrema: minimum number of local extrema
        """
        self.K = K
        self.endpoints = endpoints
        self.max_smooth_iter = max_smooth_iter
        self.max_envelope_iter = max_envelope_iter
        self.envelope_epsilon = envelope_epsilon
        self.convergence_epsilon = convergence_epsilon
        self.min_extrema = min_extrema

    def __call__(self, signal: np.ndarray, K: Optional[int] = None) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, K=K)

    def is_monotonous(self, signal: np.ndarray) -> bool:
        """Determine whether the signal is a (non-strict) monotone sequence"""
        # 该方法用于迭代循环终止条件的确定
        if len(signal) <= 0:
            # 序列长度小于0无法判断
            return True  # 调试错误
        else:
            # 返回是否单调的测试
            return self.is_monotonous_increase(signal) or self.is_monotonous_decrease(signal)

    @staticmethod
    def is_monotonous_increase(signal: np.ndarray) -> bool:
        """判断输入信号是否单调增加"""
        y0 = signal[0]  # 取第一个样本
        for y1 in signal:
            # 遍历整个信号
            if y1 < y0:
                return False
            y0 = y1
        return True

    @staticmethod
    def is_monotonous_decrease(signal: np.ndarray) -> bool:
        """判断输入信号是否单调下降"""
        y0 = signal[0]  # 取第一个样本
        for y1 in signal:
            # 遍历整个信号
            if y1 > y0:
                return False
            y0 = y1
        return True

    def find_extrema(self, signal: np.ndarray) -> np.ndarray:
        """Find all local extreme points of the signal"""
        # 通过点的数目可以判断是否还能进行进一步的分解
        n = len(signal)

        # 通过scipy的argrelextrema方法确定输入序列中的极值点
        extrema = np.append(argrelextrema(signal, np.greater)[0], argrelextrema(signal, np.less)[0])
        extrema.sort()  # 对极值点进行排序
        if self.endpoints:
            # 是否要将输入信号的两个端点看作极值点
            if extrema[0] != 0:
                extrema = np.insert(extrema, 0, 0)
            if extrema[-1] != n - 1:
                extrema = np.append(extrema, n - 1)

        return extrema

    def moving_average_smooth(self, signal: np.ndarray, window: int) -> np.ndarray:
        """对输入信号通过滑动平均的方式进行平滑处理"""

        n = len(signal)  # 输入信号的长度

        # at least one nearby sample is needed for average
        if window < 3:
            window = 3

        # adjust length of sliding window to an odd number for symmetry
        if (window % 2) == 0:
            window += 1

        half = window // 2

        # 初始化滑动平均分解的参数
        weight = np.array(list(range(1, half + 2)) + list(range(half, 0, -1)))
        assert (len(weight) == window)  # 确保参数长度为窗口的大小

        smoothed = signal
        # 开始迭代
        for _ in range(self.max_smooth_iter):
            head = list()
            tail = list()
            w_num = half
            for i in range(half):
                # 分别处理头部和尾部
                head.append(np.array([smoothed[j] for j in range(i - (half - w_num), i + half + 1)]))
                tail.append(np.flip([smoothed[-(j + 1)] for j in range(i - (half - w_num), i + half + 1)]))
                w_num -= 1

            # 通过卷积运算实现平滑
            smoothed = np.convolve(smoothed, weight, mode='same')
            smoothed[half: - half] = smoothed[half: - half] / sum(weight)

            w_num = half
            for i in range(half):
                smoothed[i] = sum(head[i] * weight[w_num:]) / sum(weight[w_num:])
                smoothed[-(i + 1)] = sum(tail[i] * weight[: - w_num]) / sum(weight[: - w_num])
                w_num -= 1
            if self.is_smooth(smoothed, n):
                # 当信号已经处理到平滑状态则停止迭代
                break
        return smoothed  # 返回平滑处理后的信号

    @staticmethod
    def is_smooth(signal: np.ndarray, n: int) -> bool:
        """Determine whether a signal is smooth"""
        for x in range(1, n):
            # 如果有两个连续不相等的点说明不光滑
            if signal[x] == signal[x - 1]:
                return False
        return True

    def local_mean_and_envelope(self, signal: np.ndarray, extrema):
        """Calculate the local mean function and local envelope function according to the location of the extreme points"""
        n = len(signal)
        k = len(extrema)
        assert (1 < k <= n)
        # construct square signal
        mean = []
        enve = []
        prev_mean = (signal[extrema[0]] + signal[extrema[1]]) / 2
        prev_enve = abs(signal[extrema[0]] - signal[extrema[1]]) / 2
        e = 1
        # 开始计算局部均值和包络谱
        # 该过程为算法的核心
        for x in range(n):
            if (x == extrema[e]) and (e + 1 < k):
                next_mean = (signal[extrema[e]] + signal[extrema[e + 1]]) / 2
                mean.append((prev_mean + next_mean) / 2)
                prev_mean = next_mean
                next_enve = abs(signal[extrema[e]] - signal[extrema[e + 1]]) / 2
                enve.append((prev_enve + next_enve) / 2)
                prev_enve = next_enve
                e += 1
            else:
                mean.append(prev_mean)
                enve.append(prev_enve)
        # smooth square signal
        window = max(np.diff(extrema)) // 3
        return np.array(mean), self.moving_average_smooth(mean, window), \
            np.array(enve), self.moving_average_smooth(enve, window)

    def extract_product_function(self, signal: np.ndarray) -> np.ndarray:
        """执行一次局部均值分解算法"""
        s = signal
        n = len(signal)
        envelopes = []  # 用于存放本次包络谱分析的结果

        def component():
            # Calculate PF，using PF_i(t) = a_i(t)* s_in()，其中a_i = a_i0 * a_i1 * ... * a_in
            c = s
            for e in envelopes:
                c = c * e
            return c

        # 开始通过迭代的方法分离包络信号
        for _ in range(self.max_envelope_iter):
            # 首先通过观察信号的极值点判断是否可以早停
            extrema = self.find_extrema(s)
            if len(extrema) <= 3:
                break
            # 获取输入信号的局部包络函数
            _m0, m, _a0, a = self.local_mean_and_envelope(s, extrema)
            for i in range(len(a)):
                if a[i] <= 0:
                    a[i] = 1 - 1e-4

            # 　subtracted from the original data.
            h = s - m
            #  amplitude demodulated by dividing a.
            t = h / a

            # Terminate processing when obtaining pure FM signal.
            err = sum(abs(1 - a)) / n
            if err <= self.envelope_epsilon:
                break
            # Terminate processing when modulation signal converges.
            err = sum(abs(s - t)) / n
            if err <= self.convergence_epsilon:
                break
            envelopes.append(a)
            s = t

        return component()

    def fit_transform(self, signal: np.ndarray, K: Optional[int] = None) -> np.ndarray:
        """
        Signal decomposition using Local Mean Decomposition (LMD) algorithm
        :param signal: the time domain signal (1D numpy array)  to be decomposed
        :param K: the maximum number of IFMs to be decomposed
        :return: IMFs
        """
        if K is not None:
            # 调整超参数
            self.K = K
        pf = []
        # until the residual function is close to a monotone function
        residue = signal[:]
        # 通过循环迭代进行分解
        while (len(pf) < self.K) and (not self.is_monotonous(residue)):
            # 确保分解数目符号要求并且剩余的残差非单调
            if len(self.find_extrema(residue)) < self.min_extrema:
                # 剩余信号的极值点数目必须大于最小的指定数目
                break
            # 获得本次局部均值分解的结果
            component = self.extract_product_function(residue)
            # 每次迭代在原信号的基础上减去被分解的部分
            residue = residue - component
            # 记录本次分解的结果
            pf.append(component)

        # 将分解剩余的残差纳入其中
        pf.append(residue)
        return np.array(pf)
