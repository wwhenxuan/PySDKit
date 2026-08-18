# -*- coding: utf-8 -*-
"""
Time Varying Filter based Empirical Mode Decomposition
"""

import warnings
import numpy as np
from numpy import ndarray
from scipy.interpolate import pchip_interpolate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pysdkit.utils import inst_freq_local
from pysdkit.utils import divide2exp

from typing import Optional, Tuple, Dict, Any, Iterable, Union

warnings.filterwarnings("ignore")


class TVF_EMD(object):
    """
    Time Varying Filter based Empirical Mode Decomposition

    Li, Heng, Zhi Li, and Wei Mo.
    "A time varying filter approach for empirical mode decomposition."
    Signal Processing 138 (2017): 146-158.

    Python code: https://github.com/stfbnc/pytvfemd/tree/master

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/63300-time-varying-filter-based-empirical-mode-decomposition-tvf-emd
    """

    def __init__(
        self,
        max_imf: Optional[int] = 10,
        thresh_bwr: Optional[float] = 0.1,
        bsp_order: Optional[int] = 26,
        min_extrema: Optional[int] = 4,
        max_iter: Optional[int] = 100,
    ) -> None:
        """
        :param max_imf: maximum number of IMF rows (last used row is residual);
            MATLAB ``tvf_emd`` default is 50
        :param thresh_bwr: instantaneous bandwidth-ratio threshold
        :param bsp_order: B-spline order (paper / MATLAB default 26)
        :param min_extrema: stop when residual extrema count falls below this
        :param max_iter: maximum inner sifting iterations per IMF
        """
        assert max_imf > 0, "max_imf must be greater than 0"
        self.max_imf = int(max_imf)
        self.thresh_bwr = float(thresh_bwr)
        self.bsp_order = int(bsp_order)
        self.min_extrema = int(min_extrema)
        self.max_iter = int(max_iter)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Time Varying Filter based Empirical Mode Decomposition (TVF_EMD)"

    @staticmethod
    def find_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get the extrema points of the input signal"""
        length = len(signal)  # Get the length of the signal
        diff = np.diff(signal)  # Get the first-order difference of the signal

        # Exclude the two endpoints
        d1 = diff[:-1]
        d2 = diff[1:]

        # Preliminary screening of extrema points through the difference sequence
        # Occurs when the signal's rate of change goes from negative to positive
        indmin = np.where((d1 * d2 < 0.0) & (d1 < 0.0))[0] + 1
        # Occurs when the signal's rate of change goes from positive to negative
        indmax = np.where((d1 * d2 < 0.0) & (d1 > 0.0))[0] + 1

        # Further process the flat parts where the difference is zero
        if len(np.where(diff == 0.0)[0]) > 0:
            # Array to store results
            imax, imin = np.array([], dtype=int), np.array([], dtype=int)

            # Mark the positions where the difference is zero in `diff`
            bad = diff == 0.0

            # Extend the array at both ends to handle the start and end
            c_bad = np.concatenate([[0], bad, [0]])

            # Used to find the start and end points of flat intervals
            dd = np.diff(c_bad)
            debs = np.where(dd == 1)[0]
            fins = np.where(dd == -1)[0]

            # Further judgment
            if len(debs) > 0 and debs[0] == 0:
                if len(debs) > 1:
                    debs = debs[1:]
                    fins = fins[1:]
                else:
                    debs = np.array([], dtype=int)
                    fins = np.array([], dtype=int)

            if len(debs) > 0:
                if fins[-1] == length - 1:
                    if len(debs) > 1:
                        debs = debs[:-1]
                        fins = fins[:-1]
                    else:
                        debs = np.array([], dtype=int)
                        fins = np.array([], dtype=int)

            if len(debs) > 0:
                for k in range(len(debs)):
                    if diff[debs[k] - 1] > 0:
                        if diff[fins[k]] < 0:
                            imax = np.concatenate(
                                [imax, [np.round((fins[k] + debs[k]) / 2)]]
                            )
                    else:
                        if diff[fins[k]] > 0:
                            imin = np.concatenate(
                                [imin, [np.round((fins[k] + debs[k]) / 2)]]
                            )

            # Merge the final indices
            if len(imax) > 0:
                indmax = np.sort(np.concatenate([indmax, imax]))
            if len(imin) > 0:
                indmin = np.sort(np.concatenate([indmin, imin]))

        return indmin.astype(int), indmax.astype(int)

    def _anti_mode_mixing(
        self,
        y: np.ndarray,
        bis_freq: np.ndarray,
        ind_remove_pad: np.ndarray,
        num_padding: int,
    ) -> Union[None, int, float, complex, ndarray, Iterable]:
        """Remove unstable parts or 'noise' from the signal based on the extrema points of the input signal and certain rules, and smooth the signal through interpolation"""
        org_bis_freq = bis_freq.copy()
        flag_intermitt = 0
        t = np.arange(0, len(bis_freq), dtype=int)
        intermitt = np.array([], dtype=int)

        # Extract the extrema points of the input signal
        indmin_y, indmax_y = self.find_extrema(y)
        zero_span = np.array([], dtype=int)

        # Handle intermittent changes in the input signal
        for i in range(1, len(indmax_y) - 1):
            time_span = np.arange(indmax_y[i - 1], indmax_y[i + 1] + 1, dtype=int)
            span_min = np.min(bis_freq[time_span])
            if span_min > 0 and (
                (np.max(bis_freq[time_span]) - span_min) / span_min > 0.25
            ):
                zero_span = np.concatenate([zero_span, time_span])
        # The values in the intermittent intervals will be set to 0
        bis_freq[zero_span] = 0

        # Calculate frequency differences
        diff_bis_freq = np.zeros(bis_freq.shape)
        for i in range(len(indmax_y) - 1):
            time_span = np.arange(indmax_y[i], indmax_y[i + 1] + 1, dtype=int)
            span_min = np.min(bis_freq[time_span])
            if span_min > 0 and (
                (np.max(bis_freq[time_span]) - span_min) / span_min > 0.25
            ):
                intermitt = np.concatenate([intermitt, [indmax_y[i]]])
                diff_bis_freq[indmax_y[i]] = (
                    bis_freq[indmax_y[i + 1]] - bis_freq[indmax_y[i]]
                )

        # Smooth the signal and fill in
        ind_remove_pad = np.delete(
            ind_remove_pad,
            np.r_[
                np.s_[0 : np.round(0.1 * len(ind_remove_pad)).astype(int)],
                np.s_[
                    np.round(0.9 * len(ind_remove_pad)).astype(int)
                    - 1 : len(ind_remove_pad)
                ],
            ],
        )

        inters = np.intersect1d(ind_remove_pad, intermitt)
        if len(inters) > 0:
            flag_intermitt = 1

        for i in range(1, len(intermitt) - 1):
            u1 = intermitt[i - 1]
            u2 = intermitt[i]
            u3 = intermitt[i + 1]
            if diff_bis_freq[u2] > 0:
                bis_freq[u1 : u2 + 1] = 0
            if diff_bis_freq[u2] < 0:
                bis_freq[u2 : u3 + 1] = 0

        temp_bis_freq = bis_freq.copy()
        temp_bis_freq[temp_bis_freq < 1e-9] = 0
        temp_bis_freq = temp_bis_freq[ind_remove_pad]
        temp_bis_freq = np.concatenate(
            [
                np.flip(temp_bis_freq[1 : 2 + num_padding - 1]),
                temp_bis_freq,
                np.flip(temp_bis_freq[-num_padding - 1 : -1]),
            ]
        )
        flip_bis_freq = np.flip(bis_freq)
        id_t = np.where(temp_bis_freq > 1e-9)[0]
        id_f = np.where(flip_bis_freq > 1e-9)[0]
        if len(id_t) > 0 and len(id_f) > 0:
            temp_bis_freq[0] = bis_freq[np.where(bis_freq > 1e-9)[0][0]]
            temp_bis_freq[-1] = flip_bis_freq[np.where(flip_bis_freq > 1e-9)[0][0]]
        else:
            temp_bis_freq[0] = bis_freq[0]
            temp_bis_freq[-1] = bis_freq[-1]

        bis_freq = temp_bis_freq.copy()
        # MATLAB returns the (unmodified) output_cutoff = original bis_freq
        if len(t[np.where(bis_freq != 0)[0]]) < 2:
            output_cutoff = org_bis_freq.copy()
            output_cutoff[output_cutoff > 0.45] = 0.45
            output_cutoff[output_cutoff < 0] = 0
            return output_cutoff

        # Signal smoothing using the pchip interpolation algorithm
        bis_freq = pchip_interpolate(
            t[np.where(bis_freq != 0)[0]], bis_freq[np.where(bis_freq != 0)[0]], t
        )
        # Eliminate large fluctuations in frequency
        flip_bis_freq = np.flip(org_bis_freq)
        if (
            len(np.where(org_bis_freq > 1e-9)[0]) > 0
            and len(np.where(flip_bis_freq > 1e-9)[0]) > 0
        ):
            org_bis_freq[0] = org_bis_freq[np.where(org_bis_freq > 1e-9)[0][0]]
            org_bis_freq[-1] = flip_bis_freq[np.where(flip_bis_freq > 1e-9)[0][0]]

        org_bis_freq[np.where(org_bis_freq < 1e-9)[0]] = 0
        org_bis_freq[0] = bis_freq[0]
        org_bis_freq[-1] = bis_freq[-1]
        org_bis_freq = pchip_interpolate(
            t[np.where(org_bis_freq != 0)[0]],
            org_bis_freq[np.where(org_bis_freq != 0)[0]],
            t,
        )

        if flag_intermitt and np.max(temp_bis_freq[ind_remove_pad]) > 1e-9:
            output_cutoff = bis_freq.copy()
        else:
            output_cutoff = org_bis_freq.copy()

        # Output frequency corrected signal
        output_cutoff[np.where(output_cutoff > 0.45)[0]] = 0.45
        output_cutoff[np.where(output_cutoff < 0)[0]] = 0

        return output_cutoff

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Execute Time Varying Filter based Empirical Mode Decomposition.

        :param signal: 1-D real array
        :return: IMF matrix ``(n_modes, n)``; the last row is the residual
            (trailing unused rows are trimmed, as in MATLAB ``tvf_emd``)
        """
        signal = np.asarray(signal, dtype=float).ravel()
        length = int(signal.size)
        imf = np.zeros((self.max_imf, length), dtype=float)
        temp_signal = signal.copy()
        n_out = 0

        for n in range(self.max_imf):
            indmin, indmax = self.find_extrema(temp_signal)

            # last allocated slot -> store residual and stop
            if n == self.max_imf - 1:
                imf[n, :] = temp_signal
                n_out = n + 1
                break

            # not enough extrema to continue
            if (indmin.size + indmax.size) < self.min_extrema:
                imf[n, :] = temp_signal
                if np.any(temp_signal != 0):
                    n_out = n + 1
                break

            num_padding = int(np.round(len(temp_signal) * 0.5))
            y = temp_signal.copy()
            flag_stopiter = False
            ind_remove_pad = np.arange(num_padding, num_padding + length, dtype=int)
            localmean = np.zeros(length + 2 * num_padding, dtype=float)

            for niter in range(self.max_iter):
                # symmetric padding (MATLAB: fliplr(y(2:num_padding+1)))
                y = np.concatenate(
                    [
                        np.flip(y[1 : num_padding + 1]),
                        y,
                        np.flip(y[-num_padding - 1 : -1]),
                    ]
                )
                ind_remove_pad = np.arange(num_padding, len(y) - num_padding, dtype=int)

                indmin_y, indmax_y = self.find_extrema(y)
                index_c_y = np.sort(np.concatenate([indmin_y, indmax_y]))

                inst_amp_0, inst_freq_0 = inst_freq_local(y)
                (
                    _a1,
                    _f1,
                    _a2,
                    _f2,
                    bis_freq,
                    inst_bwr,
                    _avg_freq,
                ) = divide2exp(y, inst_amp_0, inst_freq_0)

                inst_bwr_2 = inst_bwr.copy()
                for j in range(0, len(index_c_y) - 2, 2):
                    ind = np.arange(index_c_y[j], index_c_y[j + 2] + 1, dtype=int)
                    inst_bwr_2[ind] = np.mean(inst_bwr[ind])

                bis_freq = np.asarray(bis_freq, dtype=float).copy()
                bis_freq[inst_bwr_2 < self.thresh_bwr] = 1e-12
                bis_freq[bis_freq > 0.5] = 0.45
                bis_freq[bis_freq <= 0] = 1e-12

                # anti mode-mixing (applied twice, as in MATLAB)
                for _ in range(2):
                    bis_freq = self._anti_mode_mixing(
                        y, bis_freq, ind_remove_pad.copy(), num_padding
                    )
                    bis_freq = bis_freq[ind_remove_pad]
                    bis_freq = np.concatenate(
                        [
                            np.flip(bis_freq[1 : num_padding + 1]),
                            bis_freq,
                            np.flip(bis_freq[-num_padding - 1 : -1]),
                        ]
                    )

                # ----- stopping criteria (set flag; do not peel yet) -----
                temp_inst_bwr = inst_bwr_2[ind_remove_pad]
                # MATLAB 1-based round(..); convert to 0-based inclusive slice
                ind_start = max(int(np.round(len(temp_inst_bwr) * 0.05)) - 1, 0)
                ind_end = max(int(np.round(len(temp_inst_bwr) * 0.95)) - 1, ind_start)
                mean_bwr = float(np.mean(temp_inst_bwr[ind_start : ind_end + 1]))
                thr_iter = self.thresh_bwr + self.thresh_bwr / 4.0 * (niter + 1)

                # MATLAB: (iter>=2 && mean<thr) || iter>=6 || (nimf>1 && mean<thr)
                if (
                    (niter >= 1 and mean_bwr < thr_iter)
                    or (niter >= 5)
                    or (n > 0 and mean_bwr < thr_iter)
                ):
                    flag_stopiter = True

                frac_wide = np.count_nonzero(
                    temp_inst_bwr[ind_start : ind_end + 1] > self.thresh_bwr
                ) / max(len(inst_bwr_2[ind_remove_pad]), 1)
                if frac_wide < 0.2:
                    flag_stopiter = True

                # ----- local mean via time-varying B-spline filter -----
                phi = np.zeros(len(bis_freq), dtype=float)
                for i in range(len(bis_freq) - 1):
                    phi[i + 1] = phi[i] + 2.0 * np.pi * bis_freq[i]

                indmin_knot, indmax_knot = self.find_extrema(np.cos(phi))
                index_c_knot = np.sort(np.concatenate([indmin_knot, indmax_knot]))
                if len(index_c_knot) > 2:
                    localmean = fit_spline(
                        np.arange(0, len(y), dtype=int),
                        y,
                        index_c_knot,
                        self.bsp_order,
                    )
                else:
                    flag_stopiter = True

                if len(index_c_knot) > 2:
                    denom = np.min(np.abs(localmean[ind_remove_pad]))
                    if denom > 0 and (
                        np.max(np.abs(y[ind_remove_pad] - localmean[ind_remove_pad]))
                        / denom
                        < 1e-3
                    ):
                        flag_stopiter = True

                    temp_residual = (y - localmean)[ind_remove_pad]
                    trim_r = max(int(np.round(len(temp_residual) * 0.1)), 1)
                    temp_residual = temp_residual[trim_r:-trim_r]
                    localmean2 = localmean[ind_remove_pad]
                    trim_m = max(int(np.round(len(localmean2) * 0.1)), 1)
                    localmean2 = localmean2[trim_m:-trim_m]
                    amp_ref = np.max(np.abs(inst_amp_0[ind_remove_pad])) + 1e-30
                    if (
                        np.abs(np.max(localmean2)) / amp_ref < 3.5e-2
                        or np.abs(np.max(temp_residual)) / amp_ref < 1e-2
                    ):
                        flag_stopiter = True

                if flag_stopiter:
                    imf[n, :] = y[ind_remove_pad]
                    temp_signal = temp_signal - imf[n, :]
                    n_out = n + 1
                    break

                y = (y - localmean)[ind_remove_pad]

            else:
                # exhausted max_iter without stop flag (rare): take current residual
                imf[n, :] = y if y.size == length else y[ind_remove_pad]
                temp_signal = temp_signal - imf[n, :]
                n_out = n + 1

        # MATLAB: imf(nimf:MAX_IMF,:)=[]  — drop unused trailing rows
        if n_out == 0:
            return np.zeros((1, length), dtype=float)
        return imf[:n_out, :]


def fit_spline(x: np.ndarray, y: np.ndarray, breaks: np.ndarray, n: int) -> np.ndarray:
    """
    Try to fit the spline for the input signal
    This function comes from https://github.com/stfbnc/pytvfemd/tree/master
    """
    x, y, breaks = check_knots(x, y, breaks)
    pp_dict = spline_base(breaks, n)
    pieces = pp_dict["pieces"]
    a_sp = spline_eval(pp_dict, x)
    ibin = np.digitize(x, breaks[1:-1])

    mx = len(x)
    ii = np.vstack((ibin, np.ones((n - 1, mx)))).astype(int)
    ii = np.cumsum(ii, axis=0)
    jj = np.tile(np.arange(0, mx).astype(int), (n, 1))
    ii = np.mod(ii, pieces)
    a_sp = csr_matrix(
        (a_sp.flatten(), (ii.flatten(), jj.flatten())), shape=(pieces, mx), dtype=float
    )
    a_sp.eliminate_zeros()

    if pieces < 20 * n / np.log(1.7 * n):
        a_sp = a_sp.todense().transpose()
        u = np.linalg.lstsq(a_sp, y)[0]
    else:
        u = spsolve(a_sp * a_sp.T, a_sp * csr_matrix(y, dtype=float).T)

    jj = np.mod(np.arange(0, pieces + n - 1, dtype=int), pieces)
    u = u[jj]

    ii = np.vstack(
        (np.tile(np.arange(0, pieces, dtype=int), n), np.ones((n - 1, n * pieces)))
    )
    ii = np.cumsum(ii, axis=0)
    jj = np.tile(np.arange(0, n * pieces, dtype=int), (n, 1))
    c_mtx = csr_matrix(
        (pp_dict["coefs"].flatten("F"), (ii.flatten("F"), jj.flatten("F"))),
        shape=(pieces + n - 1, n * pieces),
        dtype=float,
    )
    coefs = u * c_mtx
    coefs = np.reshape(coefs, (int(len(coefs) / n), n), order="F")

    pp_spline = pp_struct(breaks, coefs, 1)
    sp_fit = spline_eval(pp_spline, np.arange(0, len(y), dtype=int))

    return sp_fit[0]


def check_knots(
    x: np.ndarray, y: np.ndarray, knots: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check if x points are outside knots range
    This function comes from https://github.com/stfbnc/pytvfemd/tree/master
    """
    if len(np.where(np.diff(knots) <= 0)[0]) > 0:
        knots = np.unique(knots)

    h = np.diff(knots)
    xlim1 = knots[0] - 0.01 * h[0]
    xlim2 = knots[-1] + 0.01 * h[-1]
    if x[0] < xlim1 or x[-1] > xlim2:
        p = knots[-1] - knots[0]
        x = ((x - knots[0]) % p) + knots[0]
        isort = np.argsort(x, kind="stable")
        x = x[isort]
        y = y[isort]

    return x, y, knots


def spline_base(breaks: np.ndarray, n: int) -> Dict[str, Union[int, Any, Any]]:
    """
    Generates B-spline base of order `n` for knots `breaks`
    This function comes from https://github.com/stfbnc/pytvfemd/tree/master
    """
    breaks = breaks.flatten()
    breaks0 = breaks.copy()
    h = np.diff(breaks)
    pieces = len(h)
    deg = n - 1

    if deg > 0:
        if deg <= pieces:
            hcopy = h.copy()
        else:
            hcopy = np.tile(h, (int(np.ceil(deg / pieces)),))
        hl = hcopy[-1 : -deg - 1 : -1]
        bl = breaks[0] - np.cumsum(hl)
        hr = hcopy[:deg]
        br = breaks[-1] + np.cumsum(hr)
        breaks = np.concatenate([bl[deg - 1 :: -1], breaks, br])
        h = np.diff(breaks)
        pieces = len(h)

    coefs = np.zeros((n * pieces, n), dtype=float)
    coefs[::n, 0] = 1

    ii = np.ones((deg + 1, pieces), dtype=int)
    ii[0, :] = np.linspace(0, pieces, pieces, endpoint=False, dtype=int)
    ii = np.cumsum(ii, axis=0)
    ii[np.where(ii > pieces - 1)] = pieces - 1
    hh = h[ii.flatten("F")]

    for k in range(1, n):
        for j in range(k):
            coefs[:, j] = coefs[:, j] * hh / (k - j)
        q = np.sum(coefs, axis=1)
        q = q.reshape((pieces, n)).T
        q = np.cumsum(q, axis=0)
        c0 = np.concatenate([np.zeros((1, pieces)), q[0:deg, :]])
        coefs[:, k] = c0.flatten("F")
        fmax = np.tile(q[n - 1, :], (n, 1))
        fmax = fmax.flatten("F")
        for j in range(k + 1):
            coefs[:, j] = coefs[:, j] / fmax
        coefs[0:-deg, 0 : k + 1] = coefs[0:-deg, 0 : k + 1] - coefs[n - 1 :, 0 : k + 1]
        coefs[::n, k] = 0

    scale = np.ones(hh.shape)
    for k in range(n - 1):
        scale = scale / hh
        coefs[:, n - k - 2] = scale * coefs[:, n - k - 2]

    pieces -= 2 * deg

    ii = np.ones((deg + 1, pieces), dtype=int) * deg
    ii[0, :] = n * np.arange(1, pieces + 1, dtype=int)
    ii = np.cumsum(ii, axis=0) - 1
    coefs = coefs[ii.flatten("F"), :]

    return pp_struct(breaks0, coefs, n)


def pp_struct(br: np.ndarray, cf: np.ndarray, d: int) -> Dict:
    """
    Structure for piecewise polynomial parameters
    This function comes from https://github.com/stfbnc/pytvfemd/tree/master
    """
    dlk = cf.shape[0] * cf.shape[1]
    l = len(br) - 1
    dl = d * l
    k = np.fix(dlk / dl + 100 * np.spacing(1)).astype(int)

    pp = {
        "breaks": br.reshape((1, l + 1))[0],
        "coefs": cf.reshape((dl, k)),
        "pieces": l,
        "order": k,
        "dim": d,
    }

    return pp


def spline_eval(pp: Dict, xx: np.ndarray) -> np.ndarray:
    """
    Evaluates piecewise polynomial
    This function comes from https://github.com/stfbnc/pytvfemd/tree/master
    """
    sizexx = xx.shape
    lx = np.prod([s for s in xx.shape]).astype(int)
    xs = xx.reshape((lx,))
    if len(sizexx) == 2 and sizexx[0] == 1:
        sizexx = (sizexx[1],)

    b = pp["breaks"]
    c = pp["coefs"]
    l = pp["pieces"]
    k = pp["order"]
    dd = pp["dim"]

    if lx > 0:
        index = np.digitize(xs, b[1:l])
    else:
        index = np.ones((lx,), dtype=int)

    infxs = np.where(xs == np.inf)[0]
    if len(infxs) != 0:
        index[infxs] = l
    nogoodxs = np.where(index < 0)[0]
    if len(nogoodxs) != 0:
        xs[nogoodxs] = -999
        index[nogoodxs] = 1

    xs = xs - b[index]
    d = np.prod(dd)

    if d > 1:
        xs = np.tile(xs, (d, 1)).transpose((1, 0)).reshape((d * lx,))
        index = d * (index + 1) - 1
        temp = np.arange(-d, 0).astype(int)
        arr = (
            np.tile(temp[np.newaxis].transpose(), (1, lx)) + np.tile(index, (d, 1)) + 1
        )
        index = arr.transpose((1, 0)).reshape((d * lx,))
    else:
        if len(sizexx) > 1:
            dd = np.array([])
        else:
            dd = 1

    v = c[index, 0]
    for i in range(1, k):
        v = xs * v + c[index, i]

    if len(nogoodxs) > 0 and k == 1 and l > 1:
        v = v.reshape((d, lx))
        v[:, nogoodxs] = -999
    v = np.reshape(v, (dd, sizexx[0]), order="F")

    return v
