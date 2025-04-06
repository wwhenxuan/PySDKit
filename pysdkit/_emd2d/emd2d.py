# -*- coding: utf-8 -*-
"""
Created on 2025/02/04 13:13:52
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import ndarray, dtype, signedinteger

from scipy.interpolate import SmoothBivariateSpline
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure

from typing import Optional, Tuple, Any


class EMD2D(object):
    """
    **Empirical Mode Decomposition 2D** for images data.

    Method decomposes images into 2D representations of loose Intrinsic Mode Functions (IMFs).

    Threshold values that control goodness of the decomposition:
       * `mse_thr` --- proto-IMF check whether small mean square error.
       * `mean_thr` --- proto-IMF chekc whether small mean value.

    Python code: https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD2d.py
    MATLAB code: https://www.mathworks.com/help/signal/ref/emd.html
    """

    def __init__(self, max_imfs: Optional[int] = -1, mse_thr: Optional[float] = 0.01, mean_thr: Optional[float] = 0.01,
                 max_iter: Optional[int] = 1000) -> None:
        """
        :param max_imfs: IMF number to which decomposition should be performed and Negative value means *all*.
        :param mse_thr: proto-IMF check whether small mean square error.
        :param mean_thr: proto-IMF chekc whether small mean value.
        :param max_iter: maximum number of iterations per single sifting in EMD2D
        """
        self.max_imfs = max_imfs
        self.mse_thr = mse_thr
        self.mean_thr = mean_thr
        self.fix_epochs = 0
        self.fix_epochs_h = 0

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Empirical Mode Decomposition 2D (EMD2D)"

    @staticmethod
    def find_extrema(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds extrema, both mininma and maxima, based on local maximum filter.
        Returns extrema in form of two rows, where the first and second are
        positions of x and y, respectively.

        :param image: input image of NumPy 2D ndarray.
        :return: - min_peaks: NumPy ndarray Minima positions,
                 - max_peaks: NumPy ndarray Maxima positions.
        """
        # define an 3x3 neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_min = maximum_filter(-image, footprint=neighborhood) == -image
        local_max = maximum_filter(image, footprint=neighborhood) == image

        # can't distinguish between background zero and filter zero
        background = image == 0

        # appear along the bg border (artifact of the local max filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        min_peaks = local_min ^ eroded_background
        max_peaks = local_max ^ eroded_background

        min_peaks = local_min
        max_peaks = local_max
        min_peaks[[0, -1], :] = False
        min_peaks[:, [0, -1]] = False
        max_peaks[[0, -1], :] = False
        max_peaks[:, [0, -1]] = False

        min_peaks = np.nonzero(min_peaks)
        max_peaks = np.nonzero(max_peaks)

        return min_peaks, max_peaks

    def check_proto_imf(self, proto_imf: np.ndarray, proto_imf_prev: np.ndarray, mean_env: np.ndarray) -> bool:
        """
        Check whether passed (proto) IMF is actual IMF.
        Current condition is solely based on checking whether the mean is below threshold.

        :param proto_imf: Current iteration of proto IMF.
        :param proto_imf_prev: Previous iteration of proto IMF.
        :param mean_env: Local mean computed from top and bottom envelopes.
        :return: Whether current proto IMF is actual IMF.
        """
        # TODO: Sifiting is very sensitive and subtracting const val can often flip
        #      maxima with minima in decompoisition and thus repeating above/below
        #      behaviour.
        # For now, mean_env is checked whether close to zero excluding its offset.

        if np.all(np.abs(mean_env - mean_env.mean()) < self.mean_thr):
            # if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if np.allclose(proto_imf, proto_imf_prev):
            return True

        # If IMF mean close to zero (below threshold)
        if np.mean(np.abs(proto_imf)) < self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf * proto_imf)
        if mse_proto_imf < self.mse_thr:
            return True

        return False

    def fit_transform(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs EMD on input image with 2D numpy ndarray.

        :param image: Image which will be decomposed with numpy 2D array.
        :return: Set of IMFs in form of numpy array where the first dimension relates to IMF's ordinary number.
                 NumPy 3D array of [num_imfs, Height, Width].
        """
        # 获取输入图像的最大值和最小值
        image_min, image_max = np.min(image), np.max(image)

        # 获取偏置与尺度
        offset, scale = image_min, image_max - image_min

        # 对图像进行归一化
        image_norm = (image - offset) / scale

        # 初始化一个本征模态函数
        imf = np.zeros_like(image_norm)

        # 初始化所有带分解的本征模态函数
        imfNo = 0
        IMF = np.empty((imfNo,) + image.shape)

        # 算法分解的停止标识
        notFinished = True

        while notFinished is True:
            # 开始经验模态分解的迭代算法

            # 获取当前分解的子信号
            res = image_norm - np.sum(IMF[: imfNo], axis=0)
            imf = res.copy()

            # 初始化带分解图像的包络谱中值
            mean_env = np.zeros_like(image)

            # 本轮分解的停止标识
            stop_sifting = False

            # Counters
            n = 0  # All iterations for current imf.
            n_h = 0  # counts when mean(proto_imf) < threshold

            while not stop_sifting and n < self.max_imfs:
                # Update the counter
                n += 1

                # 寻找输入待分解图像的极大值和极小值
                min_peaks, max_peaks = self.find_extrema(imf)

                # 获取输入信号的包络谱
                mean_env = (min_peaks + max_peaks) / 2

                # 记录上一次输入的数据
                imf_old = imf.copy()

                # 将原本的输入减去包络谱均值
                imf = imf - mean_env

                # Fix number of iterations
                if self.fix_epochs:
                    if n >= self.fix_epochs + 1:
                        stop_sifting = True

                # Fix number of iterations after number of zero-crossings
                # and extrema differ at most by one.
                elif self.fix_epochs_h:
                    if n == 1:
                        continue
                    if self.check_proto_imf(imf, imf_old, mean_env):
                        n_h += 1
                    else:
                        n_h = 0
