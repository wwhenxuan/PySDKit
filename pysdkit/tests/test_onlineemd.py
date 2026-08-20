# -*- coding: utf-8 -*-
"""
Unit tests for Online Empirical Mode Decomposition (Online EMD).
"""

import unittest

import numpy as np

from pysdkit import OnlineEMD
from pysdkit._emd.online_emd import OnlineEMD as ModuleOnlineEMD
from pysdkit._emd.online_emd import (
    OEMDStage,
    extrema_indices,
    extract_first_imf,
    fig2_signal,
    first_window_weights,
    matlab_colon,
    oemd_init,
    oemd_iter,
    parse_emd_algo,
    residual_stage_index,
    sliding_window_weights,
    stages_to_imfs,
    truncated_gaussian,
)
from pysdkit.data import load_oemd_ecg


def _prefix_reconstruction(stages, length: int) -> np.ndarray:
    """Sum IMF rows and the residual over a committed prefix."""
    matrix = stages_to_imfs(stages, n_samples=length, fill=0.0)
    return matrix[:, :length].sum(axis=0)


class OnlineEMDTestCase(unittest.TestCase):
    """Tests for :class:`pysdkit.OnlineEMD` and module-level helpers."""

    def setUp(self) -> None:
        toy = fig2_signal(stop=400.0, step=0.5)
        self.signal = toy["signal"]
        self.oemd = OnlineEMD(n_extrema=10, max_imfs=-1, emd_algo=2)

    def test_str(self) -> None:
        """``str(OnlineEMD)`` reports the algorithm name."""
        self.assertEqual(
            str(self.oemd),
            "Online Empirical Mode Decomposition (Online EMD)",
        )

    def test_init_stores_parameters(self) -> None:
        """Constructor stores window, IMF-cap and sifting parameters."""
        oemd = OnlineEMD(n_extrema=12, max_imfs=3, emd_algo="fixe", bound=3.0)
        self.assertEqual(oemd.n_extrema, 12)
        self.assertEqual(oemd.max_imfs, 3)
        self.assertEqual(oemd.emd_algo, 1)
        self.assertEqual(oemd.bound, 3.0)
        self.assertEqual(len(oemd.stages), 1)
        self.assertIsInstance(oemd.stages[0], OEMDStage)

    def test_import_from_package_root(self) -> None:
        """OnlineEMD is exported from the package root and ``_emd``."""
        self.assertIs(OnlineEMD, ModuleOnlineEMD)

    def test_invalid_n_extrema(self) -> None:
        """Fewer than 4 extrema per window is rejected."""
        with self.assertRaises(ValueError):
            OnlineEMD(n_extrema=3)

    def test_invalid_emd_algo(self) -> None:
        """Unknown local-EMD stopping rules raise ``ValueError``."""
        with self.assertRaises(ValueError):
            OnlineEMD(emd_algo="unknown")
        with self.assertRaises(ValueError):
            parse_emd_algo(7)

    def test_invalid_bound(self) -> None:
        """Non-positive Gaussian truncation is rejected."""
        with self.assertRaises(ValueError):
            OnlineEMD(bound=0.0)

    def test_parse_emd_algo_aliases(self) -> None:
        """Integer and string aliases map onto 0, 1, 2."""
        self.assertEqual(parse_emd_algo("rilling"), 0)
        self.assertEqual(parse_emd_algo(1), 1)
        self.assertEqual(parse_emd_algo("FIX_H"), 2)

    def test_matlab_colon(self) -> None:
        """``start:step:stop`` matches MATLAB length and endpoints."""
        samp = matlab_colon(np.pi / 2.0, 0.5, 10.0)
        self.assertGreater(samp.size, 1)
        self.assertAlmostEqual(samp[0], np.pi / 2.0)
        self.assertLessEqual(samp[-1], 10.0 + 1e-12)
        np.testing.assert_allclose(np.diff(samp), 0.5)

    def test_truncated_gaussian_properties(self) -> None:
        """Weight is max at 0 and vanishes at the truncation bound."""
        bound = 3.0
        w0 = truncated_gaussian(np.array([0.0]), bound=bound)[0]
        w_edge = truncated_gaussian(np.array([bound]), bound=bound)[0]
        w_mid = truncated_gaussian(np.array([1.0]), bound=bound)[0]
        self.assertGreater(w0, w_mid)
        self.assertGreater(w_mid, 0.0)
        self.assertAlmostEqual(w_edge, 0.0, places=12)
        self.assertGreaterEqual(
            truncated_gaussian(np.linspace(-bound, bound, 21), bound=bound).min(),
            -1e-15,
        )

    def test_extrema_sine(self) -> None:
        """A sampled sine yields alternating peaks and troughs."""
        time = np.linspace(0.0, 6.0 * np.pi, 300)
        idx = extrema_indices(np.sin(time))
        self.assertGreaterEqual(idx.size, 6)
        self.assertTrue(np.all(np.diff(idx) > 0))
        self.assertGreaterEqual(idx[0], 0)
        self.assertLess(idx[-1], time.size)

    def test_extrema_too_short(self) -> None:
        """Fewer than 3 samples have no interior extrema."""
        np.testing.assert_array_equal(
            extrema_indices([1.0, 2.0]), np.array([], dtype=int)
        )

    def test_extract_first_imf_length(self) -> None:
        """Local EMD returns an IMF the same length as the window."""
        window = np.sin(np.linspace(0.0, 8.0 * np.pi, 80))
        for algo in (0, 1, 2):
            imf = extract_first_imf(window, algo)
            self.assertEqual(imf.shape, window.shape)
            self.assertTrue(np.all(np.isfinite(imf)))

    def test_first_window_weights_shape_and_head(self) -> None:
        """First-window weights cover the last extremum and start at 1."""
        extr = np.array([2, 8, 14, 20, 26, 32, 38, 44, 50, 56], dtype=int)
        weights = first_window_weights(extr, n_extrema=10, bound=3.0)
        self.assertEqual(weights.size, int(extr[-1]) + 1)
        np.testing.assert_allclose(weights[: extr[1] + 1], 1.0)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertGreaterEqual(weights.min(), -1e-12)

    def test_sliding_window_weights_length(self) -> None:
        """Sliding weights match ``extr[l-1] - extr[0] + 1`` and start at 0."""
        extr = np.array([5, 11, 17, 23, 29, 35, 41, 47, 53, 59], dtype=int)
        tail = 4
        weights = sliding_window_weights(extr, window_tail=tail, n_extrema=10)
        self.assertEqual(weights.size, int(extr[-1] - extr[0] + 1))
        self.assertAlmostEqual(weights[0], 0.0, places=12)
        self.assertGreater(weights.max(), 0.0)

    def test_oemd_init_defaults(self) -> None:
        """``oemd_init`` builds one empty stage with the requested flags."""
        stages = oemd_init(max_imfs=-1, n_extrema=10, emd_algo=2)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].data.size, 0)
        self.assertEqual(stages[0].window_tail, 0)
        self.assertEqual(stages[0].window_head, 0)
        self.assertEqual(stages[0].max_imf, -1)

    def test_fit_transform_shape(self) -> None:
        """``fit_transform`` returns ``(K, T)`` with finite values."""
        imfs = self.oemd.fit_transform(self.signal)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], self.signal.size)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        a = OnlineEMD(n_extrema=10, emd_algo=2)(self.signal)
        b = OnlineEMD(n_extrema=10, emd_algo=2).fit_transform(self.signal)
        np.testing.assert_allclose(a, b)

    def test_prefix_reconstruction(self) -> None:
        """Committed prefix reconstructs the input (IMF sum + residual)."""
        self.oemd.fit_transform(self.signal)
        res_idx = residual_stage_index(self.oemd.stages)
        self.assertGreaterEqual(res_idx, 1)
        self.assertLess(res_idx, len(self.oemd.stages))
        prefix = int(self.oemd.stages[res_idx].data.size)
        self.assertGreater(prefix, 20)
        recon = _prefix_reconstruction(self.oemd.stages, prefix)
        np.testing.assert_allclose(recon, self.signal[:prefix], atol=1e-8, rtol=1e-8)
        n_imf1 = min(self.oemd.stages[0].imf.size, self.oemd.stages[1].data.size)
        np.testing.assert_allclose(
            self.oemd.stages[0].imf[:n_imf1] + self.oemd.stages[1].data[:n_imf1],
            self.signal[:n_imf1],
            atol=1e-8,
        )

    def test_streaming_matches_batch(self) -> None:
        """Packet-wise ``update`` matches dumping the whole record."""
        batch = OnlineEMD(n_extrema=10, emd_algo=2)
        imf_batch = batch.fit_transform(self.signal)

        stream = OnlineEMD(n_extrema=10, emd_algo=2)
        pkt = 20
        for start in range(0, self.signal.size, pkt):
            stream.update(self.signal[start : start + pkt])
        imf_stream = stream.get_imfs()

        n_rows = min(imf_batch.shape[0], imf_stream.shape[0])
        committed = min(batch.committed_length(), stream.committed_length())
        self.assertGreater(committed, 20)
        np.testing.assert_allclose(
            imf_batch[:n_rows, :committed],
            imf_stream[:n_rows, :committed],
            atol=1e-8,
        )

    def test_max_imfs_one(self) -> None:
        """``max_imfs=1`` yields one IMF plus a residual row."""
        imfs = OnlineEMD(n_extrema=10, max_imfs=1, emd_algo=2).fit_transform(
            self.signal
        )
        self.assertEqual(imfs.shape[0], 2)

    def test_max_imfs_one_residual_stage(self) -> None:
        """After ``max_imfs=1`` the second stage is the unsifted residual."""
        oemd = OnlineEMD(n_extrema=10, max_imfs=1, emd_algo=2)
        oemd.fit_transform(self.signal)
        self.assertGreaterEqual(len(oemd.stages), 2)
        self.assertEqual(oemd.stages[1].window_head, 0)
        self.assertEqual(oemd.stages[1].max_imf, 0)
        self.assertEqual(residual_stage_index(oemd.stages), 1)

    def test_emd_algo_fixe(self) -> None:
        """``emd_algo=1`` (10 siftings) still reconstructs the prefix."""
        oemd = OnlineEMD(n_extrema=10, emd_algo=1)
        oemd.fit_transform(self.signal)
        res_idx = residual_stage_index(oemd.stages)
        prefix = int(oemd.stages[res_idx].data.size)
        self.assertGreater(prefix, 10)
        recon = _prefix_reconstruction(oemd.stages, prefix)
        np.testing.assert_allclose(recon, self.signal[:prefix], atol=1e-8)

    def test_emd_algo_rilling(self) -> None:
        """``emd_algo=0`` (default / Rilling-like stop) runs and is finite."""
        imfs = OnlineEMD(n_extrema=10, emd_algo=0).fit_transform(self.signal)
        self.assertEqual(imfs.shape[1], self.signal.size)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_append_iterate_update(self) -> None:
        """``append`` + ``iterate`` matches ``update`` on the same chunk."""
        a = OnlineEMD(n_extrema=10, emd_algo=2)
        a.append(self.signal[:80])
        a.iterate()
        b = OnlineEMD(n_extrema=10, emd_algo=2)
        b.update(self.signal[:80])
        np.testing.assert_allclose(a.get_imfs(), b.get_imfs())

    def test_reset_clears_buffer(self) -> None:
        """``reset`` drops samples and IMFs from a previous run."""
        self.oemd.fit_transform(self.signal)
        self.assertGreater(self.oemd.stages[0].data.size, 0)
        self.oemd.reset()
        self.assertEqual(self.oemd.stages[0].data.size, 0)
        self.assertIsNone(self.oemd.imfs)

    def test_append_rejects_2d(self) -> None:
        """A genuine 2-D array is not a univariate stream."""
        with self.assertRaises(ValueError):
            self.oemd.append(np.ones((4, 8)))

    def test_empty_append_ignored(self) -> None:
        """Empty chunks do not change the buffer."""
        self.oemd.append([])
        self.assertEqual(self.oemd.stages[0].data.size, 0)

    def test_not_enough_extrema_returns_signal(self) -> None:
        """A short trend has no window and is returned as the residual."""
        trend = np.linspace(0.0, 1.0, 30)
        imfs = OnlineEMD(n_extrema=10, emd_algo=2).fit_transform(trend)
        self.assertEqual(imfs.shape, (1, trend.size))
        np.testing.assert_allclose(imfs[0], trend)
        self.assertEqual(OnlineEMD(n_extrema=10).committed_length(), 0)

    def test_get_imfs_and_stored_residue(self) -> None:
        """``get_imfs`` fills ``imfs`` / ``residue`` on the instance."""
        oemd = OnlineEMD(n_extrema=10, emd_algo=2)
        matrix = oemd.fit_transform(self.signal)
        self.assertIsNotNone(oemd.imfs)
        self.assertIsNotNone(oemd.residue)
        np.testing.assert_allclose(
            np.vstack([oemd.imfs, oemd.residue[None, :]]), matrix
        )

    def test_fig2_signal_matches_matlab_recipe(self) -> None:
        """Helper reproduces ``example_oemd_fig2.m`` component formulas."""
        toy = fig2_signal(stop=50.0, step=0.5)
        n_samples = toy["samp"].size
        self.assertEqual(toy["signal"].size, n_samples)
        np.testing.assert_allclose(toy["comp1"], np.sin(toy["samp"]))
        np.testing.assert_allclose(
            toy["signal"],
            toy["comp1"] + toy["comp2"] + toy["comp3"] + toy["trend"],
        )
        np.testing.assert_allclose(toy["trend"][0], 0.0)
        np.testing.assert_allclose(toy["trend"][-1], 10.0)

    def test_oemd_iter_mutates_same_list(self) -> None:
        """``oemd_iter`` returns the input list after advancing it."""
        stages = oemd_init(max_imfs=-1, n_extrema=10, emd_algo=2)
        stages[0].data = self.signal.copy()
        out = oemd_iter(stages)
        self.assertIs(out, stages)
        self.assertGreater(stages[0].window_head, 0)
        self.assertGreaterEqual(len(stages), 2)

    def test_ecg_smoke_and_prefix(self) -> None:
        """Packaged ECG (paper Fig. 5) decomposes and reconstructs a prefix."""
        record = load_oemd_ecg()
        signal = np.asarray(record["signal"], dtype=float)
        self.assertEqual(signal.size, 1280)
        self.assertEqual(record["fs"], 128.0)
        snippet = signal[:400]
        oemd = OnlineEMD(n_extrema=10, emd_algo=1)
        imfs = oemd.fit_transform(snippet)
        self.assertEqual(imfs.shape[1], snippet.size)
        self.assertGreaterEqual(imfs.shape[0], 2)
        res_idx = residual_stage_index(oemd.stages)
        prefix = int(oemd.stages[res_idx].data.size)
        self.assertGreater(prefix, 20)
        recon = _prefix_reconstruction(oemd.stages, prefix)
        np.testing.assert_allclose(recon, snippet[:prefix], atol=1e-6)

    def test_committed_length_before_run(self) -> None:
        """No samples have been committed on a fresh instance."""
        self.assertEqual(self.oemd.committed_length(), 0)


if __name__ == "__main__":
    unittest.main()
