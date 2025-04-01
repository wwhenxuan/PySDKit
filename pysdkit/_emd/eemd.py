# -*- coding: utf-8 -*-
"""
Created on 2025/02/03 18:36:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Code taken from https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
"""
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from multiprocessing import Pool

from typing import Optional, Union, Sequence, Tuple, List, Dict

from pysdkit._emd import EMD
from pysdkit.utils import get_timeline


class EEMD(object):
    """
    Ensemble Empirical Mode Decomposition

    Ensemble empirical mode decomposition (EEMD) [Wu2009] is noise-assisted technique, which is meant to be more robust
    than simple Empirical Mode Decomposition (EMD). The robustness is checked by performing many decompositions on signals slightly
    perturbed from their initial position. In the grand average over all IMF results the noise will cancel each other out and the result is pure decomposition.
    Wu, Zhaohua, and Norden E. Huang. "Ensemble empirical mode decomposition: a noise-assisted data analysis method."
    Advances in adaptive data analysis 1.01 (2009): 1-41.

    Python code: https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py

    MATLAB code: https://www.mathworks.com/help/signal/ref/emd.html
    """

    def __init__(
        self,
        ext_EMD=None,
        trials: int = 100,
        noise_width: float = 0.05,
        parallel: bool = False,
        separate_trends: bool = False,
        noise_kind: Optional[str] = "normal",
        processes: Optional[int] = None,
        max_imfs: Optional[int] = -1,
        max_iter: Optional[int] = 1000,
        random_seed: Optional[int] = 42,
    ) -> None:
        """
        :param ext_EMD: pre-defined EMD algorithms to be integrated
        :param trials: number of trials or EMD performance with added noise
        :param noise_width: standard deviation of Gaussian noise
        :param parallel: flag whether to use multiprocessing in EEMD execution.
                         Since each EMD(s+noise) is independent this should improve execution speed considerably.
                         *Note* that it's disabled by default because it's the most common problem when EEMD takes too long time to finish.
                         if you set the flag to True, make also sure to set `processes` to some reasonable value.
        :param separate_trends: Number of processes harness when executing in parallel mode.
                                The value should be between 1 and max that depends on your hardware.
        :param noise_kind: the specific type of noise the algorithm handles, choices: ["normal", "uniform"]
        :param processes: the number of multiple processes
        :param max_imfs: the maximum number of imf functions to be decomposed
                         -1 means to decompose as many imf functions as possible
        :param max_iter: maximum number of decomposition iterations
        :param random_seed: create a random seed for the random number generator
        """
        self.trials = trials
        self.noise_width = noise_width
        self.separate_trends = separate_trends

        self.parallel = parallel
        self.processes = processes
        self.max_imfs = max_imfs  # maximum number of imf
        self.max_iter = max_iter  # maximum number of iteration

        # Defining noise properties
        self.noise_list = ["normal", "uniform"]
        self.noise_kind = noise_kind

        # Creating a random number generator
        self.rng = np.random.RandomState(seed=random_seed)

        # Create the EMD algorithm to be integrated
        self.EMD = (
            EMD(max_imfs=max_imfs, random_seed=random_seed, max_iteration=self.max_iter)
            if ext_EMD is None
            else ext_EMD
        )

        # Saving imfs and residue for external references
        self.imfs = None
        self.residue = None
        self._all_imfs = {}

        # Used to update the trial
        self._signal, self._time, self._seq_len, self._scale = None, None, None, None

    def __call__(
        self,
        signal: np.ndarray,
        time: Optional[np.ndarray] = None,
        max_imfs: Optional[int] = None,
        progress: Optional[bool] = False,
    ) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal, time, max_imfs, progress)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Ensemble Empirical Mode Decomposition (EEMD)"

    def generate_noise(
        self, scale: float, size: Union[int, Sequence[int]]
    ) -> np.ndarray:
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

    def _update_trial(self, trial) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        A single trial evaluation, i.e., EMD(signal + noise).

        *Note*: Although the `trial` argument isn't used, it's needed for the (multiprocessing) map method.
        """
        # Generate noise sequence
        noise = self.generate_noise(scale=self._scale, size=self._seq_len)
        # Obtain the intrinsic mode functions (IMFs) from the decomposition
        # Add noise to the original signal here
        imfs = self._run_emd(
            signal=self._signal + noise, time=self._time, max_imfs=self.max_imfs
        )
        # Whether to separate the trend
        trend = None
        if self.separate_trends:
            imfs, trend = self.EMD.get_imfs_and_trend()

        return imfs, trend

    def _run_emd(
        self, signal: np.ndarray, time: np.ndarray, max_imfs: Optional[int] = -1
    ) -> np.ndarray:
        """
        Vanilla Empirical Mode Decomposition method

        Perform the specified EMD algorithm to obtain the corresponding signal decomposition results
        """
        return self.EMD.fit_transform(signal=signal, time=time, max_imfs=max_imfs)

    @property
    def all_imfs(self) -> Dict:
        """A dictionary with all computed IMFs per given order."""
        return self._all_imfs

    def ensemble_mean(self) -> np.ndarray:
        """Integrate the results of the ensemble EMD algorithm to obtain the mean result."""
        return np.array([imfs.mean(axis=0) for imfs in self._all_imfs.values()])

    def ensemble_std(self) -> np.ndarray:
        """Obtain the standard deviation of the ensemble EMD algorithm."""
        return np.array([imfs.std(axis=0) for imfs in self._all_imfs.values()])

    def ensemble_count(self) -> List[int]:
        """Count of IMFs observed for a given order, e.g., the 1st proto-IMF, in the entire ensemble."""
        return [len(imfs) for imfs in self._all_imfs.values()]

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal

        :return: obtained IMFs and residue through EMD
        """
        if self.imfs is None or self.residue is None:
            # If the algorithm has not been executed yet, there is no result for this decomposition.
            raise ValueError(
                "No IMF found. Please, run `fit_transform` method or its variant first."
            )
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
            raise ValueError(
                "No IMF found. Please, run `fit_transform` method or its variant first."
            )

        # Get the intrinsic mode function and residual respectively
        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        else:
            return imfs, residue

    def fit_transform(
        self,
        signal: np.ndarray,
        time: Optional[np.ndarray] = None,
        max_imfs: Optional[int] = None,
        progress: Optional[bool] = False,
    ) -> np.ndarray:
        """
        Perform Ensemble Empirical Mode Decomposition for the input signal

        :param signal: input ndarray signal to be decomposed
        :param time: time array for the signal
        :param max_imfs: The number of imf decomposed
        :param progress: the number of multiple processes
        :return: Set of ensemble IMFs produced from input signal of ndarray

        In general, these do not have to be, and most likely will not be, same as IMFs produced using EMD
        """
        # The length of the input signal
        seq_len = len(signal)

        # Generate timestamp array
        if time is None:
            time = get_timeline(seq_len, dtype=signal.dtype)

        # Maximum number of decompositions
        self.max_imfs = self.max_imfs if max_imfs is None else max_imfs

        # Maximum number of decompositions
        scale = self.noise_width * np.abs(np.max(signal) - np.min(signal))

        # Store the signal to be decomposed and use it to update the trial
        self._signal, self._time, self._seq_len = signal, time, seq_len
        self._scale = scale

        # For trial number of iterations perform EMD on a signal
        # Whether to perform signal decomposition in parallel
        if self.parallel:
            pool = Pool(processes=self.processes)
            map_pool = pool.map
        else:
            map_pool = map

        all_IMFs = map_pool(self._update_trial, range(self.trials))

        if self.parallel:
            pool.close()

        self._all_imfs = defaultdict(list)

        # Create an iterator object for the signal decomposition integration algorithm
        it = iter if not progress else lambda x: tqdm(x, desc="EEMD", total=self.trials)

        for imfs, trend in it(all_IMFs):
            # Start algorithm iteration and process the output results

            if trend is not None:
                # If `separate_trends` is set to True, trend items are processed separately
                self._all_imfs[-1].append(trend)

            for imf_num, imf in enumerate(imfs):
                # Add the decomposed intrinsic mode functions to the results
                self._all_imfs[imf_num].append(imf)

        # Convert defaultdict back to dict and explicitly rename `-1` position to be {the last value} for consistency.
        self._all_imfs = dict(self._all_imfs)

        # Process trend information separately
        if -1 in self._all_imfs:
            self._all_imfs[len(self._all_imfs)] = self._all_imfs.pop(-1)

        for imf_num in self._all_imfs.keys():
            # Perform type conversion on the decomposed intrinsic mode function
            self._all_imfs[imf_num] = np.array(self._all_imfs[imf_num])

        # Record the results of this decomposition
        self.imfs = self.ensemble_mean()
        self.residue = signal - np.sum(self.imfs, axis=0)

        return self.imfs
