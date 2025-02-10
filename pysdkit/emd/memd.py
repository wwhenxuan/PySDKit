# -*- coding: utf-8 -*-
"""
Created on 2025/02/04 13:10:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
We refactored from https://github.com/mariogrune/MEMD-Python-
"""
import numpy as np
import warnings

from sys import exit

from numpy import ndarray, dtype
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, Any


class MEMD(object):
    """
    Multivariate Empirical Mode Decomposition

    Rehman, Naveed, and Danilo P. Mandic. "Multivariate empirical mode decomposition."
    Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 466.2117 (2010): 1291-1302.

    G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode Decomposition and its Algorithms", Proc of the IEEE-EURASIP
     Workshop on Nonlinear Signal and Image Processing, NSIP-03, Grado (I), June 2003

    N. E. Huang et al., "A confidence limit for the Empirical Mode Decomposition and Hilbert spectral analysis",
     Proceedings of the Royal Society A, Vol. 459, pp. 2317-2345, 2003

    Python code: https://github.com/mariogrune/MEMD-Python-
    MATLAB code: http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm
    R code: https://rdrr.io/github/PierreMasselot/Library--emdr/man/memd.html
    """

    def __init__(
        self,
        stop_crit: Optional[str] = "stop",
        max_iter: Optional[int] = 128,
        n_dir: Optional[int] = 64,
        stop_vec: Optional[np.ndarray] = np.array([0.075, 0.75, 0.075]),
        stop_cnt: Optional[int] = 2,
    ) -> None:
        """
        Initialize the Multivariate Empirical Mode Decomposition algorithm
        Note that this algorithm requires the dimension of the input signal to be greater than or equal to 3

        :param stop_crit: The criterion for stopping the algorithm iteration, you can choose ['stop', 'fix_h']
        :param max_iter: The maximum number of iterations
        :param n_dir: The number of projections of the direction vector, generally this parameter does not need to be specified,
                      After the input signal, this parameter will be set to twice the number of input signal channels
        :param stop_vec: The stop parameter setting used when `stop_crit` is 'stop'
        :param stop_cnt: The stop parameter setting used when `stop_crit` is 'fix_h'
        """

        # Set the algorithm's stopping criterion
        if not isinstance(stop_crit, str) or (
            stop_crit != "stop" and stop_crit != "fix_h"
        ):
            exit("invalid stop_criteria. stop_criteria should be either fix_h or stop")
        self.stop_crit = stop_crit
        self.max_iter = max_iter

        # Set the direction projection vector
        if not isinstance(n_dir, int) or n_dir < 6:
            exit(
                "invalid num_dir. num_dir should be an integer greater than or equal to 6."
            )
        self.n_dir = n_dir

        # Use 'stop' as stopping criterion
        if not isinstance(stop_vec, (list, tuple, np.ndarray)) or any(
            x for x in stop_vec if not isinstance(x, (int, float, complex))
        ):
            exit(
                "invalid stop_vector. stop_vector should be a list with three elements, default is [0.75,0.75,0.75]"
            )
        self.stop_vec = stop_vec
        self.sd, self.sd2, self.tol = stop_vec[0], stop_vec[1], stop_vec[2]

        # Use 'fix_h' as stopping criterion
        if not isinstance(stop_cnt, int) or stop_cnt < 0:
            exit("invalid stop_count. stop_count should be a non-negative integer.")
        self.stop_cnt = stop_cnt

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Multivariate Empirical Mode Decomposition (MEMD)"

    def init_hammersley(self, N_dim: int) -> np.ndarray:
        """Initializations for Hammersley function"""
        base = [-self.n_dir]

        # Find the pointset for the given input signal
        if N_dim == 3:
            base.append(2)
            seq = np.zeros((self.n_dir, N_dim - 1))
            for it in range(0, N_dim - 1):
                seq[:, it] = hamm(self.n_dir, base[it])
        else:
            # Prime numbers for Hammersley sequence
            prm = nth_prime(N_dim - 1)
            for itr in range(1, N_dim):
                base.append(prm[itr - 1])
            seq = np.zeros((self.n_dir, N_dim))
            for it in range(0, N_dim):
                seq[:, it] = hamm(self.n_dir, base[it])

        return seq

    def stop_emd(self, signal: np.ndarray, seq: np.ndarray, N_dim: int) -> bool:
        """Controls whether to stop the iteration of the EMD algorithm"""
        ner = np.zeros(shape=(self.n_dir, 1))
        dir_vec = np.zeros(shape=(N_dim, 1))

        for it in range(0, self.n_dir):

            if N_dim != 3:
                # Multivariate signal (for N_dim ~=3) with hammersley sequence
                # Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
                b = 2 * seq[it, :] - 1

                # Find angles corresponding to the normalised sequence
                tht = np.arctan2(
                    np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[: N_dim - 1]
                ).transpose()

                # Find coordinates of unit direction vectors on n-sphere
                dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
                dir_vec[: N_dim - 1, 0] = np.cos(tht) * dir_vec[: N_dim - 1, 0]

            else:
                # Trivariate signal with hammersley sequence
                # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
                tt = 2 * seq[it, 0] - 1
                if tt > 1:
                    tt = 1
                elif tt < -1:
                    tt = -1

                # Normalize angle from 0 - 2*pi
                phirad = seq[it, 1] * 2 * np.pi
                st = np.sqrt(1.0 - tt * tt)

                dir_vec[0] = st * np.cos(phirad)
                dir_vec[1] = st * np.sin(phirad)
                dir_vec[2] = tt
            # Projection of input signal on nth (out of total ndir) direction vectors
            y = np.dot(signal, dir_vec)

            # Calculates the extrema of the projected signal
            indmin, indmax = local_peaks(y)

            ner[it] = len(indmin) + len(indmax)

        # Stops if the all projected signals have less than 3 extrema
        stop_flag = all(ner < 3)

        return stop_flag

    def stop(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seq: np.ndarray,
        seq_len: int,
        N_dim: int,
    ) -> Tuple[bool, np.ndarray]:
        try:
            env_mean, nem, nzm, amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, N_dim
            )
            sx = np.sqrt(np.sum(np.power(env_mean, 2), axis=1))

            if all(amp):
                # something is wrong here
                sx = sx / amp

            if not (
                (np.mean(sx > self.sd) > self.tol or any(sx > self.sd2)) or any(nem > 2)
            ):
                stop_flag = True
            else:
                stop_flag = False
        except Exception as e:
            env_mean = np.zeros(shape=(seq_len, N_dim))
            stop_flag = True

        return stop_flag, env_mean

    def fix(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seq: np.ndarray,
        seq_len: int,
        N_dim: int,
        counter: int,
    ) -> Tuple[bool, np.ndarray, int]:
        try:
            env_mean, nem, nzm, amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, N_dim
            )

            if all(np.abs(nzm - nem) > 1):
                stop_flag = False
                counter = 0
            else:
                counter += 1
                stop_flag = counter >= self.stop_cnt

        except Exception as e:
            env_mean = np.zeros(shape=(seq_len, N_dim))
            stop_flag = True

        return stop_flag, env_mean, counter

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Preform the Multivariate Empirical Mode Decomposition method.
        Please note that this method only works for signals with input dimension greater than or equal to 3
        (following the original MEMD MATLAB code)
        :param signal: the multivariate signal of numpy ndarray
        :return: the decomposed signal results
        """

        # Get the dimension and length of the input signal
        N_dim, seq_len = signal.shape

        if N_dim < 3:
            raise ValueError(
                "MEMD can only process signals with three or more input elements. If the signal to be processed does not meet the requirements, you can try using the MVMD algorithm."
            )

        # Initializations for Hammersley function
        seq = self.init_hammersley(N_dim)

        # Initialize the projection dimension by the dimension of the input signal
        self.n_dir = N_dim * 2

        # Generate a timestamp sequence array
        time = np.arange(1, seq_len + 1)

        # Channelize the input signal to decompose it
        signal = signal.transpose((1, 0))

        print(signal.shape)

        # List used to store decomposition results
        imfs = []

        # Record the number of decomposed IMFs
        n_imfs = 1

        # Counter
        nbit = 0

        # 开始进行迭代分解
        while not self.stop_emd(signal=signal, seq=seq, N_dim=N_dim):
            # Get current mode
            m = signal

            # computation of mean and stopping criterion
            if self.stop_crit == "stop":
                stop_flag, env_mean = self.stop(
                    signal=m, time=time, seq=seq, N_dim=N_dim, seq_len=seq_len
                )
            elif self.stop_crit == "fix_h":
                counter = 0
                stop_flag, env_mean, counter = self.fix(
                    signal=m,
                    time=time,
                    seq=seq,
                    seq_len=seq_len,
                    N_dim=N_dim,
                    counter=counter,
                )
            else:
                raise ValueError(
                    "Parameter `stop_crit` is set incorrectly! Please choose from 'stop' or 'fix_h'."
                )

            print(stop_flag)

            # In case the current mode is so small that machine precision can cause
            # spurious extrema to appear
            if np.max(np.abs(m)) < 1e-10 * (np.max(np.abs(signal))):
                if not stop_flag:
                    warnings.warn(
                        "emd:warning, forced stop of EMD : too small amplitude"
                    )
                else:
                    print("forced stop of EMD : too small amplitude")
                break

            # sifting loop
            while stop_flag is False and nbit < self.max_iter:
                # sifting
                m = m - env_mean

                # computation of mean and stopping criterion
                if self.stop_crit == "stop":
                    stop_flag, env_mean = self.stop(
                        signal=m, time=time, seq=seq, N_dim=N_dim, seq_len=seq_len
                    )
                else:
                    stop_flag, env_mean, counter = self.fix(
                        signal=m,
                        time=time,
                        seq=seq,
                        seq_len=seq_len,
                        N_dim=N_dim,
                        counter=counter,
                    )

                nbit = nbit + 1

                if nbit == (self.max_iter - 1) and nbit > 100:
                    warnings.warn(
                        "emd:warning, forced stop of sifting : too many iterations"
                    )

            # Record the results of this decomposition
            imfs.append(m)

            n_imfs += 1
            signal = signal - m
            nbit = 0

        # Stores the residue
        imfs.append(signal)
        imfs = np.asarray(imfs)

        return imfs


def local_peaks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima and minima in the input signal.

    :param x: 1D ndarray,
        The input signal.
    :return indmin: 1D ndarray,
        Indices of the detected local minima.
    :return indmax: 1D ndarray,
        Indices of the detected local maxima.

    Notes
    * This function detects local extrema by identifying points where the signal changes direction.
    * It handles zero values and repeated values to ensure accurate detection of extrema.
    """
    # Check if all values in the signal are close to zero
    if all(x < 1e-5):
        x = np.zeros((1, len(x)))

    # Calculate the length of the signal minus one
    m = len(x) - 1

    # Calculate the first-order difference of the signal
    dy = np.diff(x.transpose()).transpose()

    # Find indices where the difference is non-zero
    a = np.where(dy != 0)[0]

    # Identify points where the indices are not consecutive
    lm = np.where(np.diff(a) != 1)[0] + 1

    # Calculate the difference between non-consecutive indices
    d = a[lm] - a[lm - 1]

    # Adjust the indices to the middle of the non-consecutive points
    a[lm] = a[lm] - np.floor(d / 2)

    # Append the last index of the signal
    a = np.insert(a, len(a), m)

    # Extract the values at the adjusted indices
    ya = x[a]

    # Check if there are more than one extrema
    if len(ya) > 1:
        # Detect maxima
        pks_max, loc_max = peaks(ya)
        # Detect minima by inverting the signal
        pks_min, loc_min = peaks(-ya)

        # Extract the indices of the minima
        if len(pks_min) > 0:
            indmin = a[loc_min]
        else:
            indmin = np.asarray([])

        # Extract the indices of the maxima
        if len(pks_max) > 0:
            indmax = a[loc_max]
        else:
            indmax = np.asarray([])
    else:
        # If there are no extrema, return empty arrays
        indmin = np.array([])
        indmax = np.array([])

    return indmin, indmax


def peaks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima in the input signal

    :param x: 1D ndarray,
        The input signal.
    :return pks_max: 1D ndarray,
        Values of the detected local maxima.
    :return locs_max: 1D ndarray,
        Indices of the detected local maxima.

    Notes
    * This function detects local maxima by identifying points where the signal changes from increasing to decreasing.
    """
    # Compute the first-order difference of the signal
    dX = np.sign(np.diff(x.transpose())).transpose()

    # Identify points where the signal changes from increasing to decreasing
    locs_max = np.where(np.logical_and(dX[:-1] > 0, dX[1:] < 0))[0] + 1

    # Extract the values of the local maxima
    pks_max = x[locs_max]

    return pks_max, locs_max


def hamm(n: int, base: int) -> np.ndarray:
    """
    Generate a Hammersley sequence.

    :param n: int,
        The length of the sequence to generate.
    :param base: int,
        The base of the Hammersley sequence.

    :return seq: 1D ndarray,
        The generated Hammersley sequence.

    Notes
    * The Hammersley sequence is a low-discrepancy sequence used in quasi-Monte Carlo methods.
    * This function generates the sequence using a base-dependent algorithm.
    """
    # Initialize the sequence with zeros
    seq = np.zeros((1, n))

    # Check if the base is greater than 1
    if 1 < base:
        # Initialize the seed array
        seed = np.arange(1, n + 1)
        base_inv = 1 / base

        # Generate the sequence using a base-dependent algorithm
        while any(x != 0 for x in seed):
            digit = np.remainder(seed[0:n], base)
            seq = seq + digit * base_inv
            base_inv = base_inv / base
            seed = np.floor(seed / base)
    else:
        # Generate the sequence using a base-independent algorithm
        temp = np.arange(1, n + 1)
        seq = (np.remainder(temp, (-base + 1)) + 0.5) / (-base)

    return seq


def nth_prime(n: int) -> list:
    """
    Generate a list of the first n prime numbers.

    :param n: int,
        The number of prime numbers to generate.
    :return lst: list,
        A list containing the first n prime numbers.
    ------------
    * This function uses a helper function `is_prime` to check if a number is prime.
    * It iterates through natural numbers and appends primes to the list until the list length reaches n.
    """

    # Initialize the list with the first prime number
    lst = [2]

    # Iterate through natural numbers starting from 3
    for i in range(3, 104745):  # 104745 is an arbitrary upper limit for demonstration

        # Check if the current number is prime
        if is_prime(i):
            # Append the prime number to the list
            lst.append(i)

            # Check if the list length has reached the desired number of primes
            if len(lst) == n:
                # Return the list of primes
                return lst


def is_prime(x: int) -> bool:
    """
    Check if a number is prime.

    :param x: int,
        The number to check for primality.

    :return: bool,
        True if the number is prime, False otherwise.

    Notes
    * A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
    * This function checks divisibility by all numbers up to the given number.
    """

    # Handle the special case for 2, the only even prime number
    if x == 2:
        return True

    else:
        # Check divisibility by 2 and all odd numbers up to x-1

        for number in range(3, x):
            # If x is divisible by any number, it is not prime
            if x % number == 0 or x % 2 == 0:
                return False

        # If no divisors were found, x is prime
        return True


def envelope_mean(
    m: np.ndarray, t: np.ndarray, seq: np.ndarray, n_dir: int, seq_len: int, N_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the envelope spectrum and upper and lower means of the input sequence"""
    NBSYM = 2
    count = 0

    env_mean = np.zeros((len(t), N_dim))
    amp = np.zeros((len(t)))
    nem = np.zeros(n_dir)
    nzm = np.zeros(n_dir)

    dir_vec = np.zeros((N_dim, 1))
    for it in range(0, n_dir):
        if N_dim != 3:  # Multivariate signal (for N_dim ~=3) with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
            b = 2 * seq[it, :] - 1

            # Find angles corresponding to the normalised sequence
            tht = np.arctan2(
                np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[: N_dim - 1]
            ).transpose()

            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[: N_dim - 1, 0] = np.cos(tht) * dir_vec[: N_dim - 1, 0]

        else:  # Trivariate signal with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
            tt = 2 * seq[it, 0] - 1
            if tt > 1:
                tt = 1
            elif tt < -1:
                tt = -1

                # Normalize angle from 0 - 2*pi
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)

            dir_vec[0] = st * np.cos(phirad)
            dir_vec[1] = st * np.sin(phirad)
            dir_vec[2] = tt

        # Projection of input signal on nth (out of total ndir) direction vectors
        y = np.dot(m, dir_vec)

        # Calculates the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        nem[it] = len(indmin) + len(indmax)
        indzer = zero_crossings(y)
        nzm[it] = len(indzer)

        tmin, tmax, zmin, zmax, mode = boundary_conditions(
            indmin, indmax, t, y, m, NBSYM
        )

        # Calculate multidimensional envelopes using spline interpolation
        # Only done if number of extrema of the projected signal exceed 3
        if mode:
            fmin = CubicSpline(tmin, zmin, bc_type="not-a-knot")
            env_min = fmin(t)
            fmax = CubicSpline(tmax, zmax, bc_type="not-a-knot")
            env_max = fmax(t)
            amp = amp + np.sqrt(np.sum(np.power(env_max - env_min, 2), axis=1)) / 2
            env_mean = env_mean + (env_max + env_min) / 2
        else:  # if the projected signal has inadequate extrema
            count = count + 1

    if n_dir > count:
        env_mean = env_mean / (n_dir - count)
        amp = amp / (n_dir - count)
    else:
        env_mean = np.zeros((seq_len, N_dim))
        amp = np.zeros((seq_len))
        nem = np.zeros((n_dir))

    return env_mean, nem, nzm, amp


def zero_crossings(x: np.ndarray) -> np.ndarray:
    """
    Detect zero-crossings in the input signal.
    :param x: 1D ndarray, The input signal.
    :return indzer: 1D ndarray, Indices of the zero-crossings in the input signal.

    Notes
    * Zero-crossings are detected by identifying sign changes between consecutive samples.
    * If the signal contains zero values, these are also considered as zero-crossings.
    """
    # Detect zero-crossings by identifying sign changes between consecutive samples
    indzer = np.where(x[0:-1] * x[1:] < 0)[0]

    # Check if the signal contains zero values
    if any(x == 0):
        # Find indices where the signal is zero
        iz = np.where(x == 0)[0]

        # Check if there are consecutive zero values
        if any(np.diff(iz) == 1):
            # Create a boolean array indicating where the signal is zero
            zer = x == 0

            # Compute the difference to find the start and end of zero sequences
            dz = np.diff([0, zer, 0])

            # Find the start indices of zero sequences
            debz = np.where(dz == 1)[0]

            # Find the end indices of zero sequences
            finz = np.where(dz == -1)[0] - 1

            # Compute the midpoint indices of zero sequences
            indz = np.round((debz + finz) / 2)
        else:
            # If there are no consecutive zero values, use the zero indices directly
            indz = iz

        # Combine the detected zero-crossings and zero indices
        indzer = np.sort(np.concatenate((indzer, indz)))

    # Return the sorted indices of zero-crossings
    return indzer


def boundary_conditions(
    indmin: np.ndarray,
    indmax: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    nbsym: int,
) -> (
    Tuple[None, None, None, None, int]
    | Tuple[
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        ndarray[Any, dtype[Any]],
        int,
    ]
):
    """
    Handle boundary conditions for signal processing by extending the signal symmetrically.

    :param indmin: 1D ndarray,
        Indices of the signal's minima.
    :param indmax: 1D ndarray,
        Indices of the signal's maxima.
    :param t: 1D ndarray,
        Time stamps of the signal.
    :param x: 1D ndarray,
        Values of the signal.
    :param z: 2D ndarray,
        Values of the signal (possibly multi-channel).
    :param nbsym: int,
        Number of symmetric points to extend.

    :return tmin: 1D ndarray,
        Time stamps of the processed minima.
    :return tmax: 1D ndarray,
        Time stamps of the processed maxima.
    :return zmin: 2D ndarray,
        Values of the processed minima.
    :return zmax: 2D ndarray,
        Values of the processed maxima.
    :return mode: int,
        Processing mode (0 if the signal has inadequate extrema, 1 otherwise).

    Notes
    * The function ensures the signal's continuity at the boundaries by symmetrically extending the signal.
    * If the signal has inadequate extrema, the function returns mode 0.
    """

    # Calculate the length of the signal
    lx = len(x) - 1

    # Calculate the length of the maxima and minima arrays
    end_max = len(indmax) - 1
    end_min = len(indmin) - 1

    # Convert indices to integers
    indmin = indmin.astype(int)
    indmax = indmax.astype(int)

    # Check if the signal has inadequate extrema
    if len(indmin) + len(indmax) < 3:
        mode = 0  # Inadequate extrema
        tmin = tmax = zmin = zmax = None  # No processed values
        return tmin, tmax, zmin, zmax, mode
    else:
        mode = 1  # The projected signal has adequate extrema

    # Boundary conditions for interpolations
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1 : min(end_max + 1, nbsym + 1)])
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
            lsym = indmax[0]
        else:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
            lmin = np.concatenate(
                (np.flipud(indmin[: min(end_min + 1, nbsym - 1)]), ([0]))
            )
            lsym = 0
    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
            lmin = np.flipud(indmin[1 : min(end_min + 1, nbsym + 1)])
            lsym = indmin[0]
        else:
            lmax = np.concatenate(
                (np.flipud(indmax[: min(end_max + 1, nbsym - 1)]), ([0]))
            )
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
            rmin = np.flipud(indmin[max(end_min - nbsym, 0) : -1])
            rsym = indmin[-1]
        else:
            rmax = np.concatenate(
                (np.array([lx]), np.flipud(indmax[max(end_max - nbsym + 2, 0) :])),
            )
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
            rsym = lx
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym, 0) : -1])
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
            rsym = indmax[-1]
        else:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
            rmin = np.concatenate(
                (np.array([lx]), np.flipud(indmin[max(end_min - nbsym + 2, 0) :])),
            )
            rsym = lx

    # Calculate the time stamps for the symmetric extensions
    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    # Ensure the symmetric parts extend enough
    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
        else:
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
        if lsym == 1:
            exit("bug")
        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
        else:
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
        if rsym == lx:
            exit("bug")
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    # Extract the values for the symmetric extensions
    zlmax = z[lmax, :]
    zlmin = z[lmin, :]
    zrmax = z[rmax, :]
    zrmin = z[rmin, :]

    # Combine the symmetric extensions with the original extrema
    tmin = np.hstack((tlmin, t[indmin], trmin))
    tmax = np.hstack((tlmax, t[indmax], trmax))
    zmin = np.vstack((zlmin, z[indmin, :], zrmin))
    zmax = np.vstack((zlmax, z[indmax, :], zrmax))

    return tmin, tmax, zmin, zmax, mode


if __name__ == "__main__":
    memd = MEMD()

    inp = np.random.rand(5, 100)

    imf = memd.fit_transform(inp)
    print(imf.shape)

    imf_x = imf[:, 0, :]
    imf_y = imf[:, 1, :]
    imf_z = imf[:, 2, :]
