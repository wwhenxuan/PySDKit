# -*- coding: utf-8 -*-
"""
Created on 2026/08/01
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Serial-EMD (SEMD): fast multi-signal EMD via 1-D serialization.

Zhang, J., Feng, F., Marti-Puig, P., Caiafa, C. F., Sun, Z., Duan, F.,
and Solé-Casals, J. (2021).
Serial-EMD: Fast Empirical Mode Decomposition Method for Multi-dimensional
Signals Based on Serialization. Information Sciences.
https://doi.org/10.1016/j.ins.2021.09.033

Reference code: https://github.com/ffbear1993/serial-emd
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .emd import EMD


def concatenate_signals(matrix_x: np.ndarray, num_interval: int) -> np.ndarray:
    """
    Serialize multi-channel signals with smooth transition bridges.

    Follows Zhang et al. (2021) / the official ``serial-emd`` Python reference.

    :param matrix_x: Array of shape ``(M, N)`` — ``M`` samples (rows),
        ``N`` channels (columns).
    :param num_interval: Transition length ``D`` (``1 <= D < M``).
    :return: 1-D serialized signal of length ``M*N + D*(N-1)``.
    """
    matrix_x = np.asarray(matrix_x, dtype=float)
    if matrix_x.ndim != 2:
        raise ValueError("matrix_x must be a 2-D array of shape (n_samples, n_channels)")

    n_length, n_signal = matrix_x.shape
    if n_signal < 1:
        raise ValueError("matrix_x must contain at least one channel")
    if n_signal == 1:
        return matrix_x[:, 0].copy()

    d = int(num_interval)
    if d < 1 or d >= n_length:
        raise ValueError(
            f"num_interval must satisfy 1 <= D < M; got D={d}, M={n_length}"
        )

    # Heads of channels 2..N and tails of channels 1..N-1
    matrix_a = matrix_x[:d, 1:]
    matrix_b = matrix_x[-d:, :-1]

    # Ramp weights a_i = i / (D+1), i = 1..D  (endpoints 0 and 1 excluded)
    vector_a = np.linspace(0.0, 1.0, d + 2)[1:-1].reshape(-1, 1)
    vector_u = np.ones((n_signal - 1, 1))

    # Transition: flip(head_{i+1}) ⊙ a + flip(tail_i) ⊙ flip(a)
    matrix_t_a = np.flipud(matrix_a) * (vector_a @ vector_u.T)
    matrix_t_b = np.flipud(matrix_b) * (np.flipud(vector_a) @ vector_u.T)
    matrix_t = matrix_t_a + matrix_t_b

    # Append a dummy zero column so Fortran flattening yields the desired layout
    matrix_z = np.zeros((d, 1))
    matrix_t = np.concatenate([matrix_t, matrix_z], axis=1)

    # Stack transitions under the original block, then column-major vectorize
    matrix_r = np.concatenate([matrix_x, matrix_t], axis=0)
    matrix_r = matrix_r.flatten(order="F")
    return matrix_r[:-d]


def deconcatenate_imfs(
    matrix_r: np.ndarray,
    num_interval: int,
    num_signal: int,
    num_length: Optional[int] = None,
) -> np.ndarray:
    """
    Split serialized IMFs back into per-channel IMF tensors.

    :param matrix_r: Serialized IMFs of shape ``(L, K)`` (or ``(L,)`` for one IMF).
    :param num_interval: Transition length ``D`` used during concatenation.
    :param num_signal: Number of original channels ``N``.
    :param num_length: Original per-channel length ``M``.  Inferred from ``L``
        when omitted via ``L = M*N + D*(N-1)``.
    :return: Array of shape ``(M, K, N)``.
    """
    matrix_r = np.asarray(matrix_r, dtype=float)
    if matrix_r.ndim == 1:
        matrix_r = matrix_r.reshape(-1, 1)
    if matrix_r.ndim != 2:
        raise ValueError("matrix_r must have shape (L, K)")

    d = int(num_interval)
    n_signal = int(num_signal)
    if n_signal < 1:
        raise ValueError("num_signal must be >= 1")

    length_ser, num_mode = matrix_r.shape
    if n_signal == 1:
        m = length_ser if num_length is None else int(num_length)
        if m != length_ser:
            raise ValueError("num_length does not match serialized length for N=1")
        return matrix_r.reshape(m, num_mode, 1)

    if d < 1:
        raise ValueError("num_interval must be >= 1")

    if num_length is None:
        # L = M*N + D*(N-1)  ⇒  M = (L - D*(N-1)) / N
        numer = length_ser - d * (n_signal - 1)
        if numer % n_signal != 0:
            raise ValueError(
                "Cannot infer num_length from serialized IMF length; "
                "please pass num_length explicitly"
            )
        m = numer // n_signal
    else:
        m = int(num_length)

    expected = m * n_signal + d * (n_signal - 1)
    if length_ser != expected:
        raise ValueError(
            f"Serialized length {length_ser} incompatible with "
            f"M={m}, N={n_signal}, D={d} (expected {expected})"
        )

    # Pad D zeros, reshape Fortran-order to (M+D, N, K), drop transitions
    matrix_z = np.zeros((d, num_mode))
    matrix_pad = np.concatenate([matrix_r, matrix_z], axis=0)
    matrix_imf = matrix_pad.reshape([-1, n_signal, num_mode], order="F")
    matrix_imf = matrix_imf[:-d, :, :]
    return matrix_imf.transpose((0, 2, 1))


def transition_bridge(
    tail: np.ndarray, head: np.ndarray, num_interval: Optional[int] = None
) -> np.ndarray:
    """
    Build the linear cross-fade bridge between two adjacent channels.

    Useful for visualizing how SEMD constructs the transition that keeps the
    mean envelope continuous across channel joins.

    :param tail: Last ``D`` samples of channel ``i`` (or a longer suffix).
    :param head: First ``D`` samples of channel ``i+1`` (or a longer prefix).
    :param num_interval: Bridge length; defaults to ``min(len(tail), len(head))``.
    :return: Bridge segment of length ``D``.
    """
    tail = np.asarray(tail, dtype=float).ravel()
    head = np.asarray(head, dtype=float).ravel()
    d = int(num_interval) if num_interval is not None else min(len(tail), len(head))
    if d < 1 or d > len(tail) or d > len(head):
        raise ValueError("num_interval exceeds available head/tail length")

    a = np.linspace(0.0, 1.0, d + 2)[1:-1]
    return np.flipud(head[:d]) * a + np.flipud(tail[-d:]) * np.flipud(a)


class SEMD(object):
    """
    Serial Empirical Mode Decomposition (Serial-EMD / SEMD)

    Zhang et al., Information Sciences, 2021.

    SEMD concatenates multi-channel signals into one long 1-D series with
    smooth transition bridges, runs a standard 1-D EMD (or a compatible
    variant), then splits the IMFs back to each original channel.  This
    avoids expensive multivariate envelope interpolation (MEMD / BEMD)
    while reusing any existing univariate EMD backend.

    Input layout (PySDKit convention)
    ---------------------------------
    - univariate: ``(seq_len,)``
    - multivariate: ``(n_channels, seq_len)``

    Output layout
    -------------
    - univariate: ``(K, seq_len)``
    - multivariate: ``(K, seq_len, n_channels)``
    """

    def __init__(
        self,
        num_interval: Optional[int] = None,
        interval_ratio: float = 0.2,
        max_imfs: int = -1,
        emd: Optional[EMD] = None,
        **emd_kwargs,
    ) -> None:
        """
        :param num_interval: Transition length ``D``.  If ``None``, uses
            ``max(1, round(interval_ratio * seq_len))``.
        :param interval_ratio: Fraction of each channel length used for ``D``
            when ``num_interval`` is not given (paper default ≈ 0.2).
        :param max_imfs: Maximum number of IMFs forwarded to the EMD backend
            (``-1`` means no hard limit).
        :param emd: Optional pre-configured univariate decomposer.  Must
            expose ``fit_transform(signal, max_imfs=...)`` returning
            ``(K, L)``.  Defaults to :class:`pysdkit._emd.emd.EMD`.
        :param emd_kwargs: Extra keyword arguments used when constructing the
            default :class:`EMD` instance.
        """
        if num_interval is not None and int(num_interval) < 1:
            raise ValueError("num_interval must be a positive integer or None")
        if not (0.0 < float(interval_ratio) <= 1.0):
            raise ValueError("interval_ratio must lie in (0, 1]")

        self.num_interval = None if num_interval is None else int(num_interval)
        self.interval_ratio = float(interval_ratio)
        self.max_imfs = int(max_imfs)
        self.emd = emd if emd is not None else EMD(max_imfs=self.max_imfs, **emd_kwargs)

        # Cached intermediates for inspection / plotting
        self.serialized_signal: Optional[np.ndarray] = None
        self.serialized_imfs: Optional[np.ndarray] = None
        self.imfs: Optional[np.ndarray] = None
        self._last_shape: Optional[Tuple[int, int]] = None  # (N, M)
        self._last_D: Optional[int] = None

    def __str__(self) -> str:
        return "Serial Empirical Mode Decomposition (SEMD)"

    def __call__(
        self, signal: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        return self.fit_transform(signal=signal, max_imfs=max_imfs)

    def resolve_interval(self, seq_len: int) -> int:
        """Resolve the transition length ``D`` for a given channel length."""
        if self.num_interval is not None:
            d = self.num_interval
        else:
            d = max(1, int(round(self.interval_ratio * seq_len)))
        if d >= seq_len:
            d = max(1, seq_len - 1)
        return d

    def serialize(self, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Convert a PySDKit multivariate array into the serialized 1-D series.

        :param signal: ``(n_channels, seq_len)`` or ``(seq_len,)``
        :return: ``(serialized_1d, D)``
        """
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            return x.copy(), 0
        if x.ndim != 2:
            raise ValueError(
                "signal must be 1-D (seq_len,) or 2-D (n_channels, seq_len)"
            )

        n_channels, seq_len = x.shape
        d = self.resolve_interval(seq_len)
        # Paper / reference layout: time in rows, channels in columns
        serialized = concatenate_signals(x.T, d)
        return serialized, d

    def fit_transform(
        self, signal: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        """
        Decompose uni-/multi-channel signals with Serial-EMD.

        :param signal: ``(seq_len,)`` or ``(n_channels, seq_len)``
        :param max_imfs: Optional override for the maximum number of IMFs
        :return: IMFs with shape ``(K, seq_len)`` or ``(K, seq_len, n_channels)``
        """
        x = np.asarray(signal, dtype=float)
        if max_imfs is None:
            max_imfs = self.max_imfs
        # Only forward a hard cap when it is positive; otherwise let the backend decide
        emd_kwargs = {} if (max_imfs is None or int(max_imfs) < 0) else {"max_imfs": int(max_imfs)}

        if x.ndim == 1:
            imfs = self.emd.fit_transform(x, **emd_kwargs)
            self.serialized_signal = x.copy()
            self.serialized_imfs = np.asarray(imfs).T  # (L, K)
            self.imfs = np.asarray(imfs)
            self._last_shape = (1, x.size)
            self._last_D = 0
            return self.imfs

        if x.ndim != 2:
            raise ValueError(
                "signal must be 1-D (seq_len,) or 2-D (n_channels, seq_len)"
            )

        n_channels, seq_len = x.shape
        if n_channels < 1 or seq_len < 2:
            raise ValueError("Invalid multivariate signal shape")

        d = self.resolve_interval(seq_len)
        serialized = concatenate_signals(x.T, d)
        self.serialized_signal = serialized
        self._last_shape = (n_channels, seq_len)
        self._last_D = d

        # Univariate EMD on the long series → (K, L)
        ser_imfs = np.asarray(self.emd.fit_transform(serialized, **emd_kwargs))
        if ser_imfs.ndim != 2:
            raise RuntimeError("EMD backend must return a 2-D IMF array (K, L)")
        self.serialized_imfs = ser_imfs.T  # (L, K)

        # (M, K, N)
        imfs_mkn = deconcatenate_imfs(
            self.serialized_imfs,
            num_interval=d,
            num_signal=n_channels,
            num_length=seq_len,
        )
        # PySDKit multivariate layout: (K, seq_len, n_channels)
        self.imfs = np.transpose(imfs_mkn, (1, 0, 2))
        return self.imfs

    def reconstruct(self, imfs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sum IMFs to reconstruct the original channel(s).

        :param imfs: Optional IMF tensor; defaults to the last ``fit_transform`` result.
        :return: ``(seq_len,)`` or ``(n_channels, seq_len)``
        """
        if imfs is None:
            if self.imfs is None:
                raise RuntimeError("Call fit_transform before reconstruct()")
            imfs = self.imfs
        imfs = np.asarray(imfs)
        if imfs.ndim == 2:
            return np.sum(imfs, axis=0)
        if imfs.ndim == 3:
            return np.sum(imfs, axis=0).T  # (N, M)
        raise ValueError("imfs must be 2-D or 3-D")
