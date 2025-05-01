# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 10:29:05
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt

from typing import Optional, Tuple

from pysdkit.plot import plot_IMFs, plot_HilbertSpectrum
from pysdkit.utils import hilbert_real, hilbert_imaginary, hilbert_transform, hilbert_spectrum
from pysdkit import EMD, REMD, EEMD, CEEMDAN
from pysdkit._emd.hht.frequency import get_envelope_frequency


class HHT(object):
    """对输入的信号进行希尔伯特-黄变换"""

    def __init__(self,
                 algorithm: Optional[str] = "EMD",
                 max_imfs: Optional[int] = -1, ):
        self.algorithm = algorithm
        self.max_imfs = max_imfs

        self.emd = self._get_emd()

        # 存储原信号与采样频率
        self.signal, self.fs = None, None
        # 存储分解结果
        self.imfs, self.imfs_env, self.imfs_freq = None, None, None

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        return "Hilbert-Huang Transform (HHT)"

    def _get_emd(self) -> EMD | REMD | EEMD | CEEMDAN:
        """选择对应的经验模态分解算法"""
        if self.algorithm == "EMD":
            return EMD(max_imfs=self.max_imfs)
        elif self.algorithm == "REMD":
            return REMD(max_imfs=self.max_imfs)
        elif self.algorithm == "EEMD":
            return EEMD(self.max_imfs)
        elif self.algorithm == "CEEMDAN":
            return CEEMDAN(self.max_imfs)
        else:
            raise ValueError("algorithm must be EMD, REMD, EEMD or CEEMDAN!")

    def save_decompsition(self, signal: np.ndarray, fs: float,
                          imfs: np.ndarray,
                          imfs_env: np.ndarray, imfs_freq: np.ndarray) -> None:
        """
        记录本次希尔伯特黄变换的结果
        :param signal:
        :param fs:
        :param imfs_env:
        :param imfs_freq:
        :return:
        """
        self.signal = signal
        self.fs = fs
        self.imfs = imfs
        self.imfs_freq = imfs_freq
        self.imfs_env = imfs_env

    def fit_transform(self, signal: np.ndarray, fs: Optional[float] = None,
                      return_all: Optional[bool] = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Hilbert-Huang transform on the signal `x`, and return the amplitude and
    instantaneous frequency function of each intrinsic mode.

        :param signal:
        :param fs: Sampling frequencies in Hz.
        :param return_all:

        :return:
        """
        # 利用经验模态分解算法获得本征模态函数
        imfs = self.emd.fit_transform(signal)
        # 分析每一个模态的包络谱频率特征
        imfs_env, imfs_freq = get_envelope_frequency(imfs, fs=fs)

        # 保存本次希尔伯特黄变换的结果
        self.save_decompsition(signal=signal, fs=fs, imfs=imfs, imfs_env=imfs_env, imfs_freq=imfs_freq)

        if return_all is True:
            return imfs, imfs_env, imfs_freq
        return imfs

    def plot_IMFs(self,
                  signal: Optional[np.ndarray] = None,
                  imfs: Optional[np.ndarray] = None) -> plt.Figure:
        """
        对分解结果进行可视化
        :param signal: 待可视化的NumPy信号。
        :param imfs: 待可视化的信号分解得到的本征模态函数。

        :return: Matplotlib中pyplot的绘图Figure对象
        """

        # 获取原信号和分解后的结果
        signal = signal if signal is not None else self.signal
        imfs = imfs if imfs is not None else self.imfs

        # 处理异常信息
        if signal is None:
            raise ValueError("请输入希尔伯特黄变换后的结果或执行一次`fit_transform`方法记录分解结果！")

        # 对分解得到的本征模态函数进行可视化
        figure = plot_IMFs(signal, imfs, return_figure=True)

        return figure

    def hilbert_spectrum(self,
                         imfs_env: Optional[np.ndarray] = None,
                         imfs_freq: Optional[np.ndarray] = None,
                         fs: Optional[float] = None,
                         freq_lim: tuple = (0, 60),
                         time_scale: int = 1,
                         freq_res: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        :param imfs_env:
        :param imfs_freq:
        :param fs:
        :param freq_lim:
        :param time_scale:
        :param freq_res:
        :return:
        """
        imfs_env = imfs_env if imfs_env is not None else self.imfs_env
        imfs_freq = imfs_freq if imfs_freq is not None else self.imfs_freq
        fs = fs if fs is not None else self.fs

        spectrum, t, f = hilbert_spectrum(imfs_env=imfs_env,
                                          imfs_freq=imfs_freq,
                                          fs=fs,
                                          freq_lim=freq_lim,
                                          time_scale=time_scale,
                                          freq_res=freq_res)
        return spectrum, t, f

    def plot_spectrum(self,
                      imfs_env: Optional[np.ndarray] = None,
                      imfs_freq: Optional[np.ndarray] = None,
                      fs: Optional[float] = None,
                      freq_lim: tuple = (0, 60),
                      time_scale: int = 1,
                      freq_res: int = 1):
        imfs_env = imfs_env if imfs_env is not None else self.imfs_env
        imfs_freq = imfs_freq if imfs_freq is not None else self.imfs_freq

        # 获取希尔伯特谱
        spectrum, t, f = self.hilbert_spectrum(imfs_env=imfs_env,
                                               imfs_freq=imfs_freq,
                                               fs=fs,
                                               freq_lim=freq_lim,
                                               time_scale=time_scale,
                                               freq_res=freq_res)
        plot_HilbertSpectrum(spectrum, t, f)

if __name__ == '__main__':
    from pysdkit.data import test_hht

    t, s = test_hht()
    fs = 1000

    hht = HHT(max_imfs=4)

    imfs, imfs_env, imfs_freq = hht.fit_transform(s, fs=fs, return_all=True)
    plot_IMFs(s, imfs)
    hht.hilbert_spectrum()
    hht.plot_spectrum(imfs_env=imfs_env, imfs_freq=imfs_freq)
    plt.show()
