# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:22:12
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Two-dimensional Empirical Wavelet Transform (EWT2D).

This module extends the 1D Empirical Wavelet Transform of Gilles (2013) to images.
The default method is the separable **Tensor EWT**, which builds independent
1D Meyer filter banks along rows and columns from the averaged Fourier spectra
and applies them successively.  An optional **Littlewood-Paley (LP)** method
builds annular supports in the 2D Fourier plane from a radially averaged spectrum.

References
----------
Gilles, J., 2013. Empirical Wavelet Transform.
IEEE Transactions on Signal Processing, 61(16), pp.3999-4010.

Gilles, J., Tran, G. and Osher, S., 2014. 2D Empirical transforms. Wavelets,
Ridgelets, and Curvelets revisited. SIAM Journal on Imaging Sciences, 7(1),
pp.157-186.

Original MATLAB toolbox:
https://www.mathworks.com/matlabcentral/fileexchange/42141-empirical-wavelet-transforms

Original Python reference:
https://github.com/basile-h/EWT-Python  (repo/EWT-Python in this project)
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any

from pysdkit._ewt.ewt import (
    EWT_Boundaries_Detect,
    EWT_Boundaries_Completion,
    EWT_beta,
)


class EWT2D(object):
    """
    Two-dimensional Empirical Wavelet Transform with a class interface.

    The Tensor method (default) is a separable extension of 1D EWT:
      1. Average the magnitude spectrum along each axis.
      2. Detect Fourier-domain boundaries with the same detectors as 1D EWT.
      3. Build Meyer scaling / wavelet filters for rows and for columns.
      4. Filter the image first along columns, then along rows.

    The LP method builds concentric annular filters (Littlewood-Paley style)
    from a radially averaged 2D Fourier spectrum.

    Parameters
    ----------
    K : int, optional
        Maximum number of 1D supports (modes) detected along each axis
        (Tensor) or along the radial spectrum (LP).  Default is 5.
    method : {"tensor", "lp"}, optional
        2D EWT construction.  ``"tensor"`` uses rectangular Fourier supports;
        ``"lp"`` uses annular (radial) supports.  Default is ``"tensor"``.
    log : int, optional
        If 1, boundary detection works on the log-spectrum.
    detect : str, optional
        Boundary detector: ``"locmax"``, ``"locmaxmin"`` or ``"locmaxminf"``.
    completion : int, optional
        If 1, pad missing boundaries up to ``K - 1``.
    reg : str, optional
        Spectrum regularisation before detection: ``"none"``, ``"gaussian"``
        or ``"average"``.
    lengthFilter : float, optional
        Width of the regularisation filter.
    sigmaFilter : float, optional
        Standard deviation of the Gaussian regularisation filter.
    """

    def __init__(
        self,
        K: Optional[int] = 5,
        method: Optional[str] = "tensor",
        log: Optional[float] = 0,
        detect: Optional[str] = "locmax",
        completion: Optional[float] = 0,
        reg: Optional[str] = "average",
        lengthFilter: Optional[float] = 10,
        sigmaFilter: Optional[float] = 5,
    ) -> None:
        self.K = int(K)
        self.method = str(method).lower()
        self.log = log
        self.detect = detect
        self.completion = completion
        self.reg = reg
        self.lengthFilter = lengthFilter
        self.sigmaFilter = sigmaFilter

        # Cached artefacts from the last ``fit_transform`` call (used by inverse)
        self._last_method: Optional[str] = None
        self._mfb_row: Optional[np.ndarray] = None
        self._mfb_col: Optional[np.ndarray] = None
        self._mfb_lp: Optional[List[np.ndarray]] = None
        self._bounds_row: Optional[np.ndarray] = None
        self._bounds_col: Optional[np.ndarray] = None
        self._bounds_scales: Optional[np.ndarray] = None
        self._n_row: Optional[int] = None
        self._n_col: Optional[int] = None
        self._image_shape: Optional[Tuple[int, int]] = None

        if self.method not in ("tensor", "lp"):
            raise ValueError(f"method must be 'tensor' or 'lp', got '{method}'.")

    def __call__(
        self,
        image: np.ndarray,
        N: Optional[int] = None,
        return_all: Optional[bool] = False,
    ) -> Union[np.ndarray, Tuple]:
        """Allow instances to be called like functions."""
        return self.fit_transform(image=image, N=N, return_all=return_all)

    def __str__(self) -> str:
        """Return the full name and abbreviation of the algorithm."""
        return "Empirical Wavelet Transform for 2D Images (EWT2D)"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_transform(
        self,
        image: np.ndarray,
        N: Optional[int] = None,
        return_all: Optional[bool] = False,
    ) -> Union[np.ndarray, Tuple]:
        """
        Perform the 2D Empirical Wavelet Transform on an image.

        Parameters
        ----------
        image : np.ndarray
            Real-valued 2D array of shape ``(H, W)``.
        N : int, optional
            Override for ``K`` (number of supports to detect).
        return_all : bool, optional
            If True, also return filter banks and detected boundaries.

        Returns
        -------
        modes : np.ndarray
            Empirical wavelet coefficients of shape ``(n_modes, H, W)``.
            For Tensor, ``n_modes = n_row_filters * n_col_filters`` with
            row-major ordering ``mode[i * n_col + j] ↔ (row_i, col_j)``.
            For LP, ``n_modes`` equals the number of annular filters.
        extras : dict, optional
            Only when ``return_all=True``.  Contains filter banks and bounds.
        """
        image = np.asarray(image, dtype=float)
        if image.ndim != 2:
            raise ValueError(
                f"EWT2D expects a 2D image of shape (H, W), got {image.shape}."
            )

        N = self.K if N is None else int(N)
        self._image_shape = image.shape

        if self.method == "tensor":
            modes, extras = self._fit_tensor(image, N)
        else:
            modes, extras = self._fit_lp(image, N)

        if return_all:
            return modes, extras
        return modes

    def inverse_transform(
        self,
        modes: np.ndarray,
        extras: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Reconstruct an image from its EWT2D coefficients.

        Parameters
        ----------
        modes : np.ndarray
            Coefficients of shape ``(n_modes, H, W)`` returned by
            ``fit_transform``.
        extras : dict, optional
            Filter-bank dictionary from ``fit_transform(..., return_all=True)``.
            If omitted, filters cached on this instance are used.

        Returns
        -------
        recon : np.ndarray
            Reconstructed image of shape ``(H, W)``.
        """
        if extras is None:
            extras = self._cached_extras()

        method = extras.get("method", self._last_method)
        if method == "tensor":
            return _iewt2d_tensor(
                modes,
                extras["mfb_row"],
                extras["mfb_col"],
                extras["n_row"],
                extras["n_col"],
            )
        if method == "lp":
            return _iewt2d_lp(modes, extras["mfb"])
        raise ValueError(f"Unknown method for inverse transform: {method}")

    # ------------------------------------------------------------------ #
    # Tensor EWT (separable rectangular supports)
    # ------------------------------------------------------------------ #
    def _fit_tensor(
        self, image: np.ndarray, N: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Separable Tensor EWT.

        Algorithm (Gilles / EWT-Python ``ewt2dTensor``)
        ----------------------------------------------
        Columns
            1. ``F_c = FFT_col(image)``
            2. ``S_c(ω) = mean_w |F_c(ω, w)|``  (average over columns)
            3. Detect boundaries on the one-sided spectrum ``S_c``
            4. Build 1D Meyer filter bank ``Ψ_col`` of length H
            5. Filter: ``u_j = IFFT_col( Ψ_col^j · F_c )``

        Rows
            6. For each column-mode ``u_j``, take ``FFT_row(u_j)``
            7. Detect row boundaries from the mean spectrum of the *original*
               image along rows (shared filter bank for all column modes)
            8. Filter each ``u_j`` with every row filter ``Ψ_row^i``

        The resulting modes live on a rectangular tiling of the 2D Fourier
        plane:  ``Λ_{i,j} = Λ_row^i × Λ_col^j``.
        """
        H, W = image.shape

        # ---- column (vertical) analysis --------------------------------
        # FFT along axis 0; average magnitude over horizontal coordinate
        ff_col = np.fft.fft(image, axis=0)
        mean_col = np.mean(np.abs(ff_col), axis=1)
        bounds_col = self._detect_boundaries_1d(mean_col, N)
        # Use the real Meyer bank that keeps the highest band equal to 1 at π
        # (tight frame); see ``_meyer_filterbank_real``.
        mfb_col = _meyer_filterbank_real(bounds_col, H)  # (H, n_col)

        # Apply each column filter
        ewtc_col = []
        for j in range(mfb_col.shape[1]):
            # Tile 1D filter across all columns, then filter in Fourier domain
            filt = np.tile(mfb_col[:, j][:, None], (1, W))
            ewtc_col.append(np.real(np.fft.ifft(filt * ff_col, axis=0)))

        # ---- row (horizontal) analysis ---------------------------------
        ff_row = np.fft.fft(image, axis=1)
        mean_row = np.mean(np.abs(ff_row), axis=0)
        bounds_row = self._detect_boundaries_1d(mean_row, N)
        mfb_row = _meyer_filterbank_real(bounds_row, W)  # (W, n_row)

        n_row = mfb_row.shape[1]
        n_col = mfb_col.shape[1]

        # Successive row filtering of every column mode
        modes = np.zeros((n_row * n_col, H, W), dtype=float)
        for i in range(n_row):
            filt_row = np.tile(mfb_row[:, i][None, :], (H, 1))
            for j in range(n_col):
                ff = np.fft.fft(ewtc_col[j], axis=1)
                modes[i * n_col + j] = np.real(np.fft.ifft(filt_row * ff, axis=1))

        # Cache for inverse_transform
        self._last_method = "tensor"
        self._mfb_row = mfb_row
        self._mfb_col = mfb_col
        self._bounds_row = bounds_row
        self._bounds_col = bounds_col
        self._n_row = n_row
        self._n_col = n_col
        self._mfb_lp = None
        self._bounds_scales = None

        extras = {
            "method": "tensor",
            "mfb_row": mfb_row,
            "mfb_col": mfb_col,
            "bounds_row": bounds_row,
            "bounds_col": bounds_col,
            "n_row": n_row,
            "n_col": n_col,
        }
        return modes, extras

    # ------------------------------------------------------------------ #
    # Littlewood-Paley EWT (annular radial supports)
    # ------------------------------------------------------------------ #
    def _fit_lp(self, image: np.ndarray, N: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Littlewood-Paley (annular) 2D EWT.

        Algorithm
        ---------
        1. Form a radial spectrum by averaging ``|FFT2(image)|`` over angles
           (via the pseudo-polar FFT magnitude, matching the reference code).
        2. Detect scale boundaries on that 1D radial spectrum.
        3. Build annular Meyer filters (scaling disk + successive rings).
        4. Filter the image in the 2D Fourier domain:
               ``mode_k = IFFT2( Ψ_k · FFT2(image) )``.

        The supports partition the frequency plane into concentric annuli,
        which is the 2D analogue of the 1D Littlewood-Paley wavelet partition.
        """
        H, W = image.shape

        # Radial spectrum via pseudo-polar FFT (same as EWT-Python)
        ppff = _ppfft(image)
        mean_ppff = np.fft.fftshift(np.mean(np.abs(ppff), axis=1))
        # One-sided radial magnitude for boundary detection
        half = mean_ppff[len(mean_ppff) // 2 :]
        bounds_scales = self._detect_boundaries_raw(half, N)
        # Map index-space bounds to [0, π]
        bounds_scales = bounds_scales * np.pi / np.ceil(len(mean_ppff) / 2.0)

        if self.completion == 1 and len(bounds_scales) < N - 1:
            bounds_scales = EWT_Boundaries_Completion(bounds_scales, N - 1)

        mfb = _ewt2d_lp_filterbank(bounds_scales, H, W)

        ff = np.fft.fft2(image)
        modes = np.zeros((len(mfb), H, W), dtype=float)
        for k, filt in enumerate(mfb):
            modes[k] = np.real(np.fft.ifft2(filt * ff))

        self._last_method = "lp"
        self._mfb_lp = mfb
        self._bounds_scales = bounds_scales
        self._mfb_row = None
        self._mfb_col = None
        self._bounds_row = None
        self._bounds_col = None
        self._n_row = None
        self._n_col = None

        extras = {
            "method": "lp",
            "mfb": mfb,
            "bounds_scales": bounds_scales,
        }
        return modes, extras

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    def _detect_boundaries_1d(self, spectrum: np.ndarray, N: int) -> np.ndarray:
        """
        Detect Fourier boundaries on a full (two-sided) 1D spectrum.

        Uses the one-sided magnitude ``spectrum[0 : ceil(L/2)]`` and scales
        the resulting indices into ``[0, π]``, exactly as in 1D ``EWT``.
        """
        one_sided = np.abs(spectrum[: int(np.ceil(spectrum.size / 2))])
        return self._detect_boundaries_raw(one_sided, N) * np.pi / round(one_sided.size)

    def _detect_boundaries_raw(self, spectrum: np.ndarray, N: int) -> np.ndarray:
        """Run the shared 1D boundary detector (returns index-space bounds)."""
        boundaries = EWT_Boundaries_Detect(
            spectrum,
            self.log,
            self.detect,
            N,
            self.reg,
            self.lengthFilter,
            self.sigmaFilter,
        )
        if self.completion == 1 and len(boundaries) < N - 1:
            # Completion expects normalised [0, π] bounds; here we stay in
            # index space and pad uniformly up to the Nyquist bin instead.
            nyquist = float(len(spectrum) - 1)
            Nd = (N - 1) - len(boundaries)
            if Nd > 0 and len(boundaries) > 0:
                deltaw = (nyquist - boundaries[-1]) / (Nd + 1)
                for _ in range(Nd):
                    boundaries = np.append(boundaries, boundaries[-1] + deltaw)
        return boundaries

    def _cached_extras(self) -> Dict[str, Any]:
        """Build an extras dict from attributes cached by ``fit_transform``."""
        if self._last_method is None:
            raise RuntimeError(
                "No cached filters. Call fit_transform first or pass extras=."
            )
        if self._last_method == "tensor":
            return {
                "method": "tensor",
                "mfb_row": self._mfb_row,
                "mfb_col": self._mfb_col,
                "bounds_row": self._bounds_row,
                "bounds_col": self._bounds_col,
                "n_row": self._n_row,
                "n_col": self._n_col,
            }
        return {
            "method": "lp",
            "mfb": self._mfb_lp,
            "bounds_scales": self._bounds_scales,
        }


# ====================================================================== #
# Functional interface
# ====================================================================== #
def ewt2d(
    image: np.ndarray,
    K: Optional[int] = 5,
    method: Optional[str] = "tensor",
    log: Optional[float] = 0,
    detect: Optional[str] = "locmax",
    completion: Optional[float] = 0,
    reg: Optional[str] = "average",
    lengthFilter: Optional[float] = 10,
    sigmaFilter: Optional[float] = 5,
    return_all: Optional[bool] = False,
) -> Union[np.ndarray, Tuple]:
    """
    Functional interface to the 2D Empirical Wavelet Transform.

    See :class:`EWT2D` for parameter descriptions.
    """
    transform = EWT2D(
        K=K,
        method=method,
        log=log,
        detect=detect,
        completion=completion,
        reg=reg,
        lengthFilter=lengthFilter,
        sigmaFilter=sigmaFilter,
    )
    return transform.fit_transform(image=image, return_all=return_all)


# ====================================================================== #
# Real 1D Meyer filter bank (tight frame, matching EWT-Python)
# ====================================================================== #
def _meyer_filterbank_real(boundaries: np.ndarray, Nsig: int) -> np.ndarray:
    """
    Build a real Meyer / Littlewood-Paley filter bank of length ``Nsig``.

    This follows EWT-Python ``ewt_LP_Filterbank(..., real=True)`` rather than
    ``EWT_Meyer_FilterBank`` in ``ewt.py``.  The key difference is that the
    highest band (support ``[ω_N, π]``) does **not** apply an upper Meyer
    transition beyond π, so ``Σ_k |ψ̂_k(ω)|² = 1`` holds up to Nyquist and the
    Tensor / LP reconstructions form a numerically tight frame.

    Parameters
    ----------
    boundaries : np.ndarray
        Detected frequency boundaries in ``(0, π)`` (0 and π excluded).
    Nsig : int
        Filter length (= signal / image side length along the filtered axis).

    Returns
    -------
    mfb : np.ndarray
        Array of shape ``(Nsig, n_filters)``; column 0 is the scaling function.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.size == 0:
        # Degenerate case: a single all-pass filter
        return np.ones((Nsig, 1), dtype=float)

    # Transition ratio γ guaranteeing a tight frame
    gamma = 1.0
    for i in range(len(boundaries) - 1):
        r = (boundaries[i + 1] - boundaries[i]) / (boundaries[i + 1] + boundaries[i])
        if 1e-16 < r < gamma:
            gamma = r
    r = (np.pi - boundaries[-1]) / (np.pi + boundaries[-1])
    if r < gamma:
        gamma = r
    gamma *= 1.0 - 1.0 / Nsig

    # Absolute frequency grid on [0, π] after Hermitian folding
    aw = np.arange(0.0, 2.0 * np.pi - 1.0 / Nsig, 2.0 * np.pi / Nsig)
    if aw.size > Nsig:
        aw = aw[:Nsig]
    elif aw.size < Nsig:
        aw = np.linspace(0.0, 2.0 * np.pi * (1.0 - 1.0 / Nsig), Nsig)
    aw = aw.copy()
    aw[Nsig // 2 :] -= 2.0 * np.pi
    aw = np.abs(aw)

    filters = [_lp_scaling(boundaries[0], aw, gamma)]
    for i in range(1, len(boundaries)):
        filters.append(_lp_wavelet(boundaries[i - 1], boundaries[i], aw, gamma))
    filters.append(_lp_wavelet(boundaries[-1], np.pi, aw, gamma))

    return np.column_stack(filters)


def _lp_scaling(w1: float, aw: np.ndarray, gamma: float) -> np.ndarray:
    """Low-pass Meyer scaling function on the absolute-frequency grid ``aw``."""
    mbn = (1.0 - gamma) * w1
    pbn = (1.0 + gamma) * w1
    an = 1.0 / (2.0 * gamma * w1)

    yms = (aw <= mbn).astype(float)
    transition = (aw > mbn) & (aw <= pbn)
    beta_vals = np.array([EWT_beta(an * (v - mbn)) for v in aw[transition]])
    yms[transition] = np.cos(np.pi * beta_vals / 2.0)
    return yms


def _lp_wavelet(wn: float, wm: float, aw: np.ndarray, gamma: float) -> np.ndarray:
    """
    Band-pass Meyer wavelet on ``[wn, wm]``.

    When ``wm == π`` (highest band), the upper transition is omitted so that
    the filter equals 1 up to Nyquist — required for a tight frame on even-
    length signals (EWT-Python ``ewt_LP_Wavelet``).
    """
    a_n = 1 if wn > np.pi else 0
    mbn = wn - gamma * abs(wn - a_n * 2.0 * np.pi)
    pbn = wn + gamma * abs(wn - a_n * 2.0 * np.pi)
    an = 1.0 / (2.0 * gamma * abs(wn - a_n * 2.0 * np.pi))

    a_m = 1 if wm > np.pi else 0
    mbm = wm - gamma * abs(wm - a_m * 2.0 * np.pi)
    pbm = wm + gamma * abs(wm - a_m * 2.0 * np.pi)
    am = 1.0 / (2.0 * gamma * abs(wm - a_m * 2.0 * np.pi))

    ymw = ((aw > mbn) & (aw < pbm)).astype(float)

    lower = (aw > mbn) & (aw < pbn)
    beta_vals = np.array([EWT_beta(an * (v - mbn)) for v in aw[lower]])
    ymw[lower] *= np.sin(np.pi * beta_vals / 2.0)

    # Only taper the upper edge when the support does not already end at π
    if wm < np.pi:
        upper = (aw > mbm) & (aw < pbm)
        beta_vals = np.array([EWT_beta(am * (v - mbm)) for v in aw[upper]])
        ymw[upper] *= np.cos(np.pi * beta_vals / 2.0)

    return ymw


# ====================================================================== #
# Inverse transforms
# ====================================================================== #
def _iewt2d_tensor(
    modes: np.ndarray,
    mfb_row: np.ndarray,
    mfb_col: np.ndarray,
    n_row: int,
    n_col: int,
) -> np.ndarray:
    """
    Inverse Tensor EWT.

    Because the Meyer filter bank forms a tight frame, reconstruction is
    obtained by re-applying the same filters and summing all contributions
    (see Gilles 2013, Sec. III; EWT-Python ``iewt2dTensor``).
    """
    _, H, W = modes.shape
    # Undo row filtering: sum over row-filters for each column mode
    ewt_col = []
    for j in range(n_col):
        acc = np.zeros((H, W), dtype=float)
        for i in range(n_row):
            ff = np.fft.fft(modes[i * n_col + j], axis=1)
            filt_row = np.tile(mfb_row[:, i][None, :], (H, 1))
            acc += np.real(np.fft.ifft(filt_row * ff, axis=1))
        ewt_col.append(acc)

    # Undo column filtering
    recon = np.zeros((H, W), dtype=float)
    for j in range(n_col):
        ff = np.fft.fft(ewt_col[j], axis=0)
        filt_col = np.tile(mfb_col[:, j][:, None], (1, W))
        recon += np.real(np.fft.ifft(filt_col * ff, axis=0))
    return recon


def _iewt2d_lp(modes: np.ndarray, mfb: List[np.ndarray]) -> np.ndarray:
    """
    Inverse Littlewood-Paley EWT.

    Reconstruction: ``image = IFFT2( Σ_k FFT2(mode_k) · Ψ_k )``.
    """
    recon_hat = np.zeros_like(np.fft.fft2(modes[0]), dtype=complex)
    for k, filt in enumerate(mfb):
        recon_hat += np.fft.fft2(modes[k]) * filt
    return np.real(np.fft.ifft2(recon_hat))


# ====================================================================== #
# LP filter-bank construction (annuli in the 2D Fourier plane)
# ====================================================================== #
def _ewt2d_lp_filterbank(bounds_scales: np.ndarray, h: int, w: int) -> List[np.ndarray]:
    """
    Build Littlewood-Paley annular filters of size ``(h, w)``.

    Following Gilles et al. and EWT-Python ``ewt2d_LPFilterbank``:

    * Extend even dimensions by one so that filters are centred on a pixel.
    * Compute the transition ratio ``γ`` that guarantees a tight frame:
          ``γ < min_n (ω_{n+1} - ω_n) / (ω_{n+1} + ω_n)``.
    * Scaling function: disk ``|ω| ≤ ω_0`` with a Meyer transition.
    * Wavelets: annuli ``[ω_n, ω_{n+1}]`` with Meyer transitions at both edges.
    * Highest band extends to ``2π``.
    * ``ifftshift`` so that filters align with NumPy's unshifted FFT layout.
    """
    h_ext = 0
    w_ext = 0
    if h % 2 == 0:
        h += 1
        h_ext = 1
    if w % 2 == 0:
        w += 1
        w_ext = 1

    # --- γ that guarantees a tight frame --------------------------------
    gamma = np.pi
    for k in range(len(bounds_scales) - 1):
        r = (bounds_scales[k + 1] - bounds_scales[k]) / (
            bounds_scales[k + 1] + bounds_scales[k]
        )
        if 1e-16 < r < gamma:
            gamma = r
    r = (np.pi - bounds_scales[-1]) / (np.pi + bounds_scales[-1])
    if 1e-16 < r < gamma:
        gamma = r
    if gamma > bounds_scales[0]:
        gamma = float(bounds_scales[0])
    gamma *= 1.0 - 1.0 / max(h, w)

    # --- radius map (distance from Fourier centre, in [0, ~π√2]) --------
    radii = np.zeros((h, w))
    h_c = h // 2 + 1
    w_c = w // 2 + 1
    for i in range(h):
        for j in range(w):
            ri = (i + 1.0 - h_c) * np.pi / h_c
            rj = (j + 1.0 - w_c) * np.pi / w_c
            radii[i, j] = np.sqrt(ri**2 + rj**2)

    mfb: List[np.ndarray] = [_ewt2d_lp_scaling(radii, bounds_scales[0], gamma)]
    for i in range(len(bounds_scales) - 1):
        mfb.append(
            _ewt2d_lp_wavelet(radii, bounds_scales[i], bounds_scales[i + 1], gamma)
        )
    mfb.append(_ewt2d_lp_wavelet(radii, bounds_scales[-1], 2.0 * np.pi, gamma))

    # Trim extension and undo fftshift centering
    if h_ext:
        h -= 1
        mfb = [f[:-1, :] for f in mfb]
    if w_ext:
        w -= 1
        mfb = [f[:, :-1] for f in mfb]
    mfb = [np.fft.ifftshift(f) for f in mfb]

    # Re-symmetrise the highest band for even-sized images (tight frame)
    if h_ext:
        _resymmetrise_highest_band_h(mfb, h, w)
    if w_ext:
        _resymmetrise_highest_band_w(mfb, h, w)

    return mfb


def _ewt2d_lp_scaling(radii: np.ndarray, bound0: float, gamma: float) -> np.ndarray:
    """
    Empirical LP scaling function (low-pass disk).

    φ̂(ω) = 1                          if |ω| ≤ (1-γ)ω₀
          = cos(π/2 β(·))              if (1-γ)ω₀ ≤ |ω| ≤ (1+γ)ω₀
          = 0                          otherwise

    where β is the Meyer auxiliary polynomial (see ``EWT_beta``).
    """
    an = 1.0 / (2.0 * gamma * bound0)
    mbn = (1.0 - gamma) * bound0
    pbn = (1.0 + gamma) * bound0

    scaling = np.zeros_like(radii)
    scaling[radii < mbn] = 1.0
    transition = (radii >= mbn) & (radii <= pbn)
    # Vectorised Meyer transition (EWT_beta is scalar; apply element-wise)
    vals = radii[transition]
    beta_vals = np.array([EWT_beta(an * (v - mbn)) for v in vals])
    scaling[transition] = np.cos(np.pi * beta_vals / 2.0)
    scaling[radii > pbn] = 0.0
    return scaling


def _ewt2d_lp_wavelet(
    radii: np.ndarray, bound1: float, bound2: float, gamma: float
) -> np.ndarray:
    """
    Empirical LP wavelet (band-pass annulus between ``bound1`` and ``bound2``).

    ψ̂_n(ω) = 1                                  if (1+γ)ω_n ≤ |ω| ≤ (1-γ)ω_{n+1}
            = sin(π/2 β(·)) on the lower rim
            = cos(π/2 β(·)) on the upper rim
            = 0                                  otherwise
    """
    wan = 1.0 / (2.0 * gamma * bound1)
    wam = 1.0 / (2.0 * gamma * bound2)
    wmbn = (1.0 - gamma) * bound1
    wpbn = (1.0 + gamma) * bound1
    wmbm = (1.0 - gamma) * bound2
    wpbm = (1.0 + gamma) * bound2

    wavelet = np.zeros_like(radii)
    inside = (radii > wmbn) & (radii < wpbm)
    wavelet[inside] = 1.0

    upper = inside & (radii >= wmbm) & (radii <= wpbm)
    vals = radii[upper]
    beta_vals = np.array([EWT_beta(wam * (v - wmbm)) for v in vals])
    wavelet[upper] *= np.cos(np.pi * beta_vals / 2.0)

    lower = inside & (radii >= wmbn) & (radii <= wpbn)
    vals = radii[lower]
    beta_vals = np.array([EWT_beta(wan * (v - wmbn)) for v in vals])
    wavelet[lower] *= np.sin(np.pi * beta_vals / 2.0)

    return wavelet


def _resymmetrise_highest_band_h(mfb: List[np.ndarray], h: int, w: int) -> None:
    """Restore Hermitian symmetry of the highest LP band when H was even."""
    s = np.zeros(w)
    band = mfb[-1]
    if w % 2 == 0:
        band[h // 2, 1 : w // 2] += band[h // 2, -1 : w // 2 : -1]
        band[h // 2, w // 2 + 1 :] = band[h // 2, w // 2 - 1 : 0 : -1]
        s += band[h // 2, :] ** 2
        band[h // 2, 1 : w // 2] /= np.sqrt(s[1 : w // 2])
        band[h // 2, w // 2 + 1 :] /= np.sqrt(s[w // 2 + 1 :])
    else:
        band[h // 2, 0 : w // 2] += band[h // 2, -1 : w // 2 : -1]
        band[h // 2, w // 2 + 1 :] = band[h // 2, w // 2 - 1 :: -1]
        s += band[h // 2, :] ** 2
        band[h // 2, 0 : w // 2] /= np.sqrt(s[0 : w // 2])
        band[h // 2, w // 2 + 1 :] /= np.sqrt(s[w // 2 + 1 :])


def _resymmetrise_highest_band_w(mfb: List[np.ndarray], h: int, w: int) -> None:
    """Restore Hermitian symmetry of the highest LP band when W was even."""
    s = np.zeros(h)
    band = mfb[-1]
    if h % 2 == 0:
        band[1 : h // 2, w // 2] += band[-1 : h // 2 : -1, w // 2]
        band[h // 2 + 1 :, w // 2] = band[h // 2 - 1 : 0 : -1, w // 2]
        s += band[:, w // 2] ** 2
        band[1 : h // 2, w // 2] /= np.sqrt(s[1 : h // 2])
        band[h // 2 + 1 :, w // 2] /= np.sqrt(s[h // 2 + 1 :])
    else:
        band[0 : h // 2, w // 2] += band[-1 : h // 2 : -1, w // 2]
        band[h // 2 + 1 :, w // 2] = band[h // 2 - 1 :: -1, w // 2]
        s += band[:, w // 2] ** 2
        # Match reference code (no sqrt on odd-H / even-W branch)
        band[0 : h // 2, w // 2] /= s[0 : h // 2]
        band[h // 2 + 1 :, w // 2] /= s[h // 2 + 1 :]


# ====================================================================== #
# Pseudo-polar FFT utilities (for LP radial spectrum)
# Ported from EWT-Python / Elad's MATLAB code, used only for LP detection.
# ====================================================================== #
def _ppfft(f: np.ndarray) -> np.ndarray:
    """
    Pseudo-polar fast Fourier transform of a 2D image.

    Samples the Fourier transform on a pseudo-polar grid (concentric squares),
    which makes radial averaging for scale detection more natural than a
    Cartesian FFT.  See Averbuch et al. / Elad's PPFFT implementation.
    """
    h, w = f.shape
    f2 = f
    N = h
    if h != w or h % 2 == 1:
        N = int(np.ceil(max(h, w) / 2.0) * 2)
        f2 = np.zeros((N, N), dtype=float)
        y0 = N // 2 - int(h / 2)
        x0 = N // 2 - int(w / 2)
        f2[y0 : y0 + h, x0 : x0 + w] = f

    ppff = np.zeros((2 * N, 2 * N), dtype=complex)

    # Quadrants 1 and 3
    ff = np.fft.fft(f2, N * 2, axis=0)
    ff = np.fft.fftshift(ff, 0)
    for i in range(-N, N):
        ppff[i + N, N - 1 :: -1] = _fracfft(ff[i + N, :], i / (N**2), centered=1)

    # Quadrants 2 and 4
    ff = np.fft.fft(f2, N * 2, axis=1)
    ff = np.fft.fftshift(ff, 1).T
    for i in range(-N, N):
        x = np.arange(0, N)
        factor = np.exp(1j * 2 * np.pi * x * (N / 2 - 1) * i / (N**2))
        ppff[i + N, N : 2 * N] = _fracfft(ff[i + N, :] * factor, i / (N**2))

    return ppff


def _fracfft(f: np.ndarray, alpha: float, centered: int = 0) -> np.ndarray:
    """Fractional FFT used inside the pseudo-polar FFT."""
    f = np.reshape(f.T, f.size)
    N = len(f)

    if centered == 1:
        x = np.arange(0, N)
        f = f * np.exp(1j * np.pi * x * N * alpha)

    x = np.append(np.arange(0, N), np.arange(-N, 0))
    factor = np.exp(-1j * np.pi * alpha * x**2)
    ff = np.append(f, np.zeros(N)) * factor
    XX = np.fft.fft(ff)
    YY = np.fft.fft(np.conj(factor))
    result = np.fft.ifft(XX * YY) * factor
    return result[0:N]
