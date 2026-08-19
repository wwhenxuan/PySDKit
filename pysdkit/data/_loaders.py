# -*- coding: utf-8 -*-
"""
Load packaged demo arrays and related example files.

All ``.npy`` assets live under ``pysdkit/data``. Algorithm modules should
import these helpers from here (or from ``pysdkit.data``) rather than
resolving files themselves.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from ._assets import data_file

_MM_CACHE: Optional[np.ndarray] = None


def _load_float1d(name: str) -> np.ndarray:
    return np.load(data_file(name)).astype(np.float64).ravel()


def _load_complex_npy(name: str) -> np.ndarray:
    return np.asarray(np.load(data_file(name)))


def load_prefixed_filter() -> np.ndarray:
    """Load the prefixed double filter used by ALIF / Iterative Filtering."""
    global _MM_CACHE
    if _MM_CACHE is None:
        _MM_CACHE = _load_float1d("prefixed_double_filter.npy")
    return _MM_CACHE


def load_imd_input_sig() -> Dict[str, np.ndarray]:
    """Load the packaged MATLAB ``InputSig`` demo (Fs = 12800 Hz, 2 s)."""
    signal = _load_float1d("input_sig.npy")
    fs = 12800.0
    t = np.arange(signal.size, dtype=float) / fs
    return {"signal": signal, "fs": fs, "t": t}


def load_imd_gearbox_snippet() -> Dict[str, np.ndarray]:
    """
    Load a short packaged snippet of the MCC5-THU gearbox CSV
    (``gearbox_vibration_x``, first 4096 samples, Fs = 12800 Hz).
    """
    signal = _load_float1d("gearbox_fault_snippet.npy")
    fs = 12800.0
    t = np.arange(signal.size, dtype=float) / fs
    return {"signal": signal, "fs": fs, "t": t}


def load_vme_ecg_055m() -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the packaged MIMIC record ``055m`` shipped with MATLAB VME.

    The File Exchange archive stores ``val`` as ``int16`` of shape
    ``(7, 7500)``.  Following ``VME_test_script.m``, channel 0 is the ECG
    and the last channel is the simultaneous reference respiration.
    Sampling rate is 125 Hz (paper: 4000 samples = 32 s).

    :return: dict with ``val``, ``ecg``, ``respiration``, ``fs``, ``t``
    """
    val = np.asarray(np.load(data_file("ecg_055m.npy")), dtype=np.float64)
    if val.ndim != 2 or val.shape[0] < 1:
        raise ValueError("ecg_055m.npy must have shape (n_channels, n_samples)")
    fs = 125.0
    t = np.arange(val.shape[1], dtype=float) / fs
    return {
        "val": val,
        "ecg": val[0].copy(),
        "respiration": val[-1].copy(),
        "fs": fs,
        "t": t,
    }


def load_dual_signal_noise() -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the packaged dual-component noisy complex demo
    (MATLAB ``Dual_signal_noise.mat``).

    Sampling rate is ``fs = 3000`` Hz and length is 3000 samples (1 s),
    matching ``test1.m``.
    """
    signal = _load_complex_npy("dual_signal_noise.npy").astype(np.complex128).ravel()
    fs = 3000.0
    t = np.arange(1, signal.size + 1, dtype=float) / fs
    return {"signal": signal, "fs": fs, "t": t, "K": 2}


def load_single_nsignal() -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the packaged single-component noisy micro-Doppler demo
    (MATLAB ``Single_nsignal.mat``).

    Sampling rate is ``fs = 8011`` Hz and length is 8011 samples (1 s),
    matching ``test2.m``.
    """
    signal = _load_complex_npy("single_nsignal.npy").astype(np.complex128).ravel()
    fs = 8011.0
    t = np.arange(signal.size, dtype=float) / fs
    return {"signal": signal, "fs": fs, "t": t, "K": 1}


def load_map2() -> np.ndarray:
    """Load the packaged MATLAB ``map2`` colormap (shape ``(64, 3)``)."""
    return np.asarray(np.load(data_file("map2.npy")), dtype=float)


def load_wind_demo(path: str) -> Dict[str, np.ndarray]:
    """
    Load a wind-demo CSV/TXT file (column 0 = wind series, ``dt=0.05`` s).

    :param path: path to a comma-separated file whose first column is the series
    """
    data = np.loadtxt(path, delimiter=",")
    y = data[:, 0]
    dt = 0.05
    t = np.arange(y.size, dtype=float) * dt
    return {"t": t, "signal": y, "dt": dt}


def test_fmd() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load the official MATLAB FMD demo signal shipped with PySDKit.

    Source: File Exchange demo ``x.mat`` accompanying
    Miao et al., IEEE Trans. Ind. Electron., 2022 (FMD).
    Sampling rate is 20 kHz; length is 20001 samples (~1 s).

    :return: ``(t, x, fs)`` time axis, signal, and sampling frequency
    """
    x = _load_float1d("fmd_demo.npy")
    fs = 2.0e4
    t = np.arange(x.size, dtype=float) / fs
    return t, x, fs


def _load_memd_multichannel(name: str) -> Dict[str, np.ndarray]:
    """Load a packaged MEMD demo array stored as ``(n_channels, n_samples)``."""
    signal = np.asarray(np.load(data_file(name)), dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("{} must be a 2-D array".format(name))
    t = np.arange(signal.shape[1], dtype=float)
    return {"signal": signal, "t": t}


def load_memd_syn_12channel() -> Dict[str, np.ndarray]:
    """
    Load the packaged MATLAB ``syn_12channel_inp.mat`` demo (``s12``).

    Twelve-channel synthetic mixture of five tones plus noise on some
    channels, length 1001, from the MEMD toolbox of Rehman & Mandic.
    """
    return _load_memd_multichannel("memd_syn_12channel.npy")


def load_memd_syn_16channel() -> Dict[str, np.ndarray]:
    """
    Load the packaged MATLAB ``syn_16channel_inp.mat`` demo (``s16``).

    Sixteen-channel synthetic mixture of six tones plus noise on some
    channels, length 1001, from the MEMD toolbox of Rehman & Mandic.
    """
    return _load_memd_multichannel("memd_syn_16channel.npy")


def load_memd_syn_hex() -> Dict[str, np.ndarray]:
    """
    Load the packaged MATLAB ``syn_hex_inp.mat`` demo (``s6``).

    Hexavariate synthetic series used in Rehman & Mandic, Proc. R. Soc. A
    466:1291–1302 (2010), Figure 3 (four sinusoids with a common scale).
    """
    return _load_memd_multichannel("memd_syn_hex.npy")


def load_memd_taichi_hex() -> Dict[str, np.ndarray]:
    """
    Load the packaged MATLAB ``taichi_hex_inp.mat`` demo (``taichi_final``).

    Real-world hexavariate Tai-chi recording: two 3-D inertial body-sensor
    streams (left wrist and left ankle) stacked into six channels of length
    800. Stored as ``(6, 800)``.
    """
    return _load_memd_multichannel("memd_taichi_hex.npy")


def test_vmd() -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load the packaged VMD demo signal (``vmd_example.npy``).

    This is the multi-component example previously stored as
    ``examples/example.npy`` (length 1024). A unit sampling rate
    ``fs = N`` is used so frequency axes are in cycles-per-record units.

    :return: ``(t, x, fs)`` time axis, signal, and sampling frequency
    """
    x = _load_float1d("vmd_example.npy")
    fs = float(x.size)
    t = np.arange(x.size, dtype=float) / fs
    return t, x, fs
