# -*- coding: utf-8 -*-
"""
Created on 2025/02/04 15:19:08
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from typing import List, Optional, Sequence, Tuple, Union

from pysdkit.emd import EMD
from pysdkit.utils import get_timeline


class CEEMDAN(object):
    """
    Complete Ensemble Empirical Mode Decomposition with Adaptive Noise

    M. E. Torres, M. A. Colominas, G. Schlotthauer and P. Flandrin,
    "A complete ensemble empirical mode decomposition with adaptive noise,"
    2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),

    M.A. Colominas, G. Schlotthauer, M.E. Torres,
    Improved complete ensemble EMD: A suitable tool for biomedical signal processing,
    In Biomed. Sig. Proc. and Control, V. 14, 2014, pp. 19--29
    Prague, Czech Republic, 2011, pp. 4144-4147, doi: 10.1109/ICASSP.2011.5947265.

    Python code: https://github.com/mariogrune/MEMD-Python-
    MATLAB code: http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm
    Word "complete" presumably refers to decomposing completely everything, even added perturbation (noise).
    """

    def __init__(self, ext_EMD=None, max_imfs: int = -1, trials: int = 100, epsilon: float = 0.005,
                 parallel: bool = False, noise_scale: float = 1.0, noise_kind: Optional[str] = 'normal',
                 range_thr: Optional[float] = 0.01, total_power_thr: Optional[float] = 0.05,
                 beta_progress: Optional[bool] = True, processes: Optional[int] = None,
                 max_iter: Optional[int] = 1000, random_seed: Optional[int] = 42) -> None:
        """
        :param ext_EMD: pre-defined EMD algorithms to be integrated
        :param max_imfs: the maximum number of imf functions to be decomposed
                         -1 means to decompose as many imf functions as possible
        :param trials: number of trials or EMD performance with added noise
        :param epsilon: scale for added noise
        :param parallel: flag whether to use multiprocessing in EEMD execution.
                         Since each EMD(s+noise) is independent this should improve execution speed considerably.
                         *Note* that it's disabled by default because it's the most common problem when EEMD takes too long time to finish.
                         if you set the flag to True, make also sure to set `processes` to some reasonable value.
        :param noise_scale: scale (amplitude) of the added noise
        :param noise_kind: the type of noise to add. Allowed are "normal" (default) and "uniform"
        :param range_thr: Range threshold used as an IMF check. The value is in percentage compared to initial signal's amplitude.
                          If absolute amplitude (max - min) is below
                          the `range_thr` then the decomposition is finished.
        :param total_power_thr: signal's power threshold. Finishes decomposition if sum(abs(r)) < thr.
        :param beta_progress: flag whether to scale all noise IMFs by their 1st IMF's standard deviation
        :param processes: the number of multiple processes
        :param max_iter: maximum number of decomposition iterations
        :param random_seed: create a random seed for the random number generator
        """
        # EMD算法的固定配置
        self.max_imfs = max_imfs
        self.trials = trials
        self.epsilon = epsilon
        self.max_iter = max_iter

        # 与多进程相关的配置参数
        self.parallel = parallel
        self.processes = processes

        # 创建噪声的参数
        self.noise_scale = noise_scale
        self.noise_kind = noise_kind
        self.noise_list = ["normal", "uniform"]
        # 是否对分解的噪声进行标准化
        self.beta_progress = beta_progress

        # 与算法停止有关的阈值
        self.range_thr = range_thr
        self.total_power_thr = total_power_thr

        # Create the EMD algorithm to be integrated
        self.EMD = EMD(max_imfs=max_imfs, random_seed=random_seed,
                       max_iteration=self.max_iter) if ext_EMD is None else ext_EMD

        # 创建随机数生成器
        self.rng = np.random.RandomState(seed=random_seed)

        # 生成的噪声序列
        self.all_noises = None

        # 存放用于分解的噪声的列表
        self.all_noise_EMD = []

        # 记录本次算法分解的结果
        self.imfs = None
        self.residue = None

        # Used to update the trial
        self._signal, self._time, self._seq_len, self._scale = None, None, None, None

    def __call__(self, signal: np.ndarray, time: Optional[np.ndarray] = None, max_imfs: Optional[int] = None,
                 progress: bool = False) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, time=time,
                                  max_imfs=max_imfs, progress=progress)

    def generate_noise(self, scale: float, size: Union[int, Sequence[int]]) -> np.ndarray:
        """
        Generate noise with specified standard deviation and size.

        The choice of noise is in ["normal", "uniform"],
        where normal has a standard deviation equal to scale and uniform has a range of [-scale / 2, scale / 2].

        :param scale: The width for the noise distribution.
        :param size: The size of the noise ndarray.
        :return: The generated white noise ndarray.
        """
        if self.noise_kind == "normal":
            # Add normal noise with mean 0
            noise = self.rng.normal(loc=0, scale=scale, size=size)
        elif self.noise_kind == "uniform":
            # Add uniform noise
            noise = self.rng.uniform(low=-scale / 2, high=scale / 2, size=size)
        else:
            raise ValueError("Unknown noise kind '{}'".format(self.noise_kind))
        return noise

    def _decompose_noise(self) -> List[np.ndarray]:
        """Decompose the noise sequences."""
        # Perform signal decomposition
        if self.parallel:
            pool = Pool(processes=self.processes)
            all_noise_EMD = pool.map(self.EMD, self.all_noises)
            pool.close()
        else:
            all_noise_EMD = [self.EMD(noise, max_imfs=-1) for noise in self.all_noises]

        # Normalize the decomposed noise sequences
        if self.beta_progress:
            all_stds = [np.std(imfs[0]) for imfs in all_noise_EMD]
            all_noise_EMD = [imfs / imfs_std for (imfs, imfs_std) in zip(all_noise_EMD, all_stds)]

        return all_noise_EMD

    def _run_eemd(self, signal: np.ndarray, time: Optional[np.ndarray] = None,
                  max_imfs: Optional[int] = None, progress: Optional[bool] = True) -> np.ndarray:
        """Perform the specified EEMD algorithm to obtain the corresponding signal decomposition results."""
        # Length of the signal
        seq_len = len(signal)

        # Generate the time array
        if time is None:
            time = get_timeline(seq_len, dtype=signal.dtype)

        # Store the current signal state
        self._signal, self._time, self._seq_len = signal, time, seq_len

        if max_imfs is not None:
            self.max_imfs = max_imfs

        # For the specified number of trials, perform EMD on the signal with added white noise
        if self.parallel:
            pool = Pool(processes=self.processes)
            map_pool = pool.imap_unordered
        else:  # Not parallel
            map_pool = map

        # Create an array to store the decomposition results
        self.imfs = np.zeros((1, seq_len))

        # Create an iterator for the decomposition process
        it = iter if not progress else lambda x: tqdm(x, desc="Decomposing noise", total=self.trials)

        # Perform signal decomposition and store the results
        for IMFs in it(map_pool(self._update_trial, range(self.trials))):
            if self.imfs.shape[0] < IMFs.shape[0]:
                num_new_layers = IMFs.shape[0] - self.imfs.shape[0]
                self.imfs = np.vstack((self.imfs, np.zeros(shape=(num_new_layers, seq_len))))
            self.imfs[: IMFs.shape[0]] += IMFs

        if self.parallel:
            pool.close()

        return self.imfs / self.trials

    def _update_trial(self, trial: int) -> np.ndarray:
        """A single trial evaluation, i.e., EMD(signal + noise)."""
        # Generate noise
        noise = self.epsilon * self.all_noise_EMD[trial][0]

        # Return the result of a single EMD execution
        return self.emd(self._signal + noise, self._time, self.max_imfs)

    def emd(self, signal: np.ndarray, time: Optional[np.ndarray] = None, max_imfs: Optional[int] = -1) -> np.ndarray:
        """
        Vanilla Empirical Mode Decomposition method
        Perform the specified EMD algorithm to obtain the corresponding signal decomposition results.
        """
        return self.EMD.fit_transform(signal=signal, time=time, max_imfs=max_imfs)

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal
        :return: obtained IMFs and residue through EMD
        """
        if self.imfs is None or self.residue is None:
            # If the algorithm has not been executed yet, there is no result for this decomposition.
            raise ValueError("No IMF found. Please, run `fit_transform` method or its variant first.")
        return self.imfs, self.residue

    def get_imfs_and_trend(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and trend from recently analysed signal.
        Note that this may differ from the `get_imfs_and_residue` as the trend isn't
        necessarily the residue. Residue is a point-wise difference between input signal
        and all obtained components, whereas trend is the slowest component (can be zero).
        :return: obtained IMFs and main trend through EMD
        """
        if self.imfs is None or self.residue is None:
            # There is no decomposition result for this storage yet
            raise ValueError("No IMF found. Please, run `fit_transform` method or its variant first.")

        # Get the intrinsic mode function and residual respectively
        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        else:
            return imfs, residue

    def end_condition(self, signal: np.ndarray, cIMFs: np.ndarray, max_imf: int) -> bool:
        """
        Test for end condition for CEEMDAN method.
        The algorithm's performance can be enhanced by adjusting the decomposition parameters.
        :param signal: the original input signal as a numpy ndarray
        :param cIMFs: the signal decomposition results
        :param max_imf: the maximum number of IMFs obtained
        :return: a boolean stop flag indicating whether to stop the CEEMDAN method
        """
        # Number of IMFs currently decomposed
        imf_number = cIMFs.shape[0]

        # Check if the maximum number of cIMFs has been reached
        if 0 < max_imf <= imf_number:
            return True

        # Compute the Empirical Mode Decomposition (EMD) for the residue
        R = signal - np.sum(cIMFs, axis=0)
        _test_imfs = self.emd(signal=R, time=None, max_imfs=1)

        # Check if the residue is an IMF or has no extrema
        if _test_imfs.shape[0] == 1:
            print("Not enough extrema")
            return True

        # Check for range threshold
        if np.max(R) - np.min(R) < self.range_thr:
            print("Finished by Range")
            return True

        # Check for power threshold
        if np.sum(np.abs(R)) < self.total_power_thr:
            print("Finished by Power")
            return True

        # The algorithm did not stop
        return False

    def fit_transform(self, signal: np.ndarray, time: Optional[np.ndarray] = None, max_imfs: Optional[int] = None,
                      progress: bool = False) -> np.ndarray:
        """
        Perform the CEEMDAN method for signal decomposition.
        :param signal: the original signal on which CEEMDAN is to be performed
        :param time: the time array of the original input signal
        :param max_imfs: the maximum number of components to extract
        :param progress: whether to print out '.' every 1s to indicate progress
        :return: the intrinsic mode functions (IMFs) after signal decomposition
        """
        # Normalize the signal's amplitude
        scale_s = np.std(signal)
        signal = signal / scale_s

        if max_imfs is not None:
            self.max_imfs = max_imfs

        # Define the noise sequences to be added
        self.all_noises = self.generate_noise(self.noise_scale, size=(self.trials, signal.size))

        # Decompose all noise and remember the standard deviation of the 1st IMF
        self.all_noise_EMD = self._decompose_noise()

        # Create the first IMF
        last_imf = self._run_eemd(signal, time, max_imfs=1, progress=progress)[0]
        res = np.empty(signal.size)

        all_cimfs = last_imf.reshape((-1, last_imf.size))
        prev_res = signal - last_imf

        # Begin executing the specific decomposition algorithm
        total = (max_imfs - 1) if max_imfs != -1 else None
        # Create an iterator object for signal decomposition
        it = iter if not progress else lambda x: tqdm(x, desc="cIMF decomposition", total=total)

        # Begin the algorithm's iteration
        for _ in it(range(self.max_imfs)):

            # Number of IMFs currently decomposed
            imf_number = all_cimfs.shape[0]

            beta = self.epsilon * np.std(prev_res)

            local_mean = np.zeros(signal.size)

            for trial in range(self.trials):
                # Skip if noise[trial] didn't have k'th mode
                noise_imf = self.all_noise_EMD[trial]
                res = prev_res.copy()
                if len(noise_imf) > imf_number:
                    res += beta * noise_imf[imf_number]

                # Extract the local mean, which is at the 2nd position
                imfs = self.emd(res, time, max_imfs=1)
                local_mean += imfs[-1] / self.trials

            # Record the results of this decomposition
            last_imf = prev_res - local_mean
            all_cimfs = np.vstack((all_cimfs, last_imf))
            prev_res = local_mean.copy()

            # Determine whether the decomposition algorithm should stop iterating
            if self.end_condition(signal=signal, cIMFs=all_cimfs, max_imf=self.max_imfs):
                # Reached the stopping condition
                print("End Decomposition")
                break

        # Clear all IMF noise
        del self.all_noise_EMD[:]

        # Record the results of this decomposition
        self.imfs = all_cimfs
        self.residue = signal * scale_s - np.sum(self.imfs, axis=0)

        return all_cimfs
