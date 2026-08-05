# -*- coding: utf-8 -*-
"""
Created on 2025/08/05
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

import numpy as np

from pysdkit import IVGNMD, ivgnmd
from pysdkit._gdmd.ivgnmd import (
    atffc_ivgnmd,
    make_ivgnmd_demo_signal,
    mtdc_ridge,
    se,
    tfsc,
    tfst,
    tfptd,
    voa_ivgnmd,
    zhang_suen_thin,
)


class IVGNMDTest(unittest.TestCase):
    """Unit tests for Improved VGNMD (IVGNMD)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.demo = make_ivgnmd_demo_signal(samp_freq=1000.0, noise_std=0.0)
        cls.signal = cls.demo["signal"]
        cls.fs = float(cls.demo["fs"][0])
        cls.t = cls.demo["t"]
        cls.true = cls.demo["modes_true"]

    def test_demo_signal_shape(self) -> None:
        self.assertEqual(self.signal.ndim, 1)
        self.assertEqual(self.true.shape, (4, self.signal.size))
        self.assertEqual(self.signal.size, 1000)

    def test_tfptd_endpoint(self) -> None:
        img = np.zeros((7, 7))
        img[3, 1:5] = 1.0
        t_mid, _ = tfptd(img, 3, 2)
        t_end, a = tfptd(img, 3, 1)
        self.assertEqual(t_mid, 2)
        self.assertEqual(t_end, 1)
        self.assertEqual(a.shape, (5, 8))

    def test_zhang_suen_thin(self) -> None:
        blob = np.zeros((20, 20))
        blob[5:15, 8:12] = 1.0
        thin = zhang_suen_thin(blob)
        self.assertTrue(np.any(thin))
        self.assertLess(int(thin.sum()), int(blob.sum()))

    def test_atffc_binary(self) -> None:
        i_spec, f = atffc_ivgnmd(self.signal, self.fs)
        self.assertEqual(i_spec.ndim, 2)
        self.assertEqual(f.ndim, 1)
        self.assertTrue(np.all((i_spec == 0) | (i_spec == 1)))
        self.assertGreater(int(i_spec.sum()), 50)

    def test_skeleton_pipeline_smoke(self) -> None:
        i_spec, _ = atffc_ivgnmd(self.signal, self.fs)
        sk = se(i_spec, h=40)
        tfc = tfsc(sk)
        paths, k = tfst(tfc, min_pixels=50, min_path_len=40)
        self.assertTrue(np.all(np.isfinite(sk)))
        self.assertGreaterEqual(int((sk > 0).sum()), 1)
        self.assertGreaterEqual(k, 1)
        self.assertEqual(len(paths), k)
        for p in paths:
            self.assertEqual(p.shape[0], 2)
            self.assertGreaterEqual(p.shape[1], 40)

    def test_mtdc_chirp_vs_dispersive(self) -> None:
        # horizontal path → chirp
        tv = np.arange(50)
        fv = 100 + (tv / 10.0).astype(int)
        rv = np.vstack([tv, fv])
        spec = np.zeros((200, 200))
        mtype, ridge = mtdc_ridge(rv, spec)
        self.assertEqual(mtype, 1)
        self.assertEqual(ridge.shape[1], 2)
        # vertical path → dispersive
        fv2 = np.arange(50)
        tv2 = np.full(50, 80)
        rv2 = np.vstack([tv2, fv2])
        mtype2, ridge2 = mtdc_ridge(rv2, spec)
        self.assertEqual(mtype2, 2)
        self.assertEqual(ridge2.shape[1], 2)

    def test_voa_recovers_chirp(self) -> None:
        i_spec, f = atffc_ivgnmd(self.signal, self.fs)
        if1 = self.demo["if1"]
        indext = np.arange(self.signal.size)
        indexf = np.clip(np.searchsorted(f, if1), 0, f.size - 1)
        ridge = np.column_stack([indext, indexf])
        mt, mf, eif = voa_ivgnmd(self.signal, 1, ridge, self.t, f, max_iter=80)
        corr = float(np.corrcoef(self.true[0], mt)[0, 1])
        self.assertGreater(abs(corr), 0.9)
        self.assertEqual(mt.size, self.signal.size)
        self.assertEqual(mf.size, f.size)

    def test_fit_transform_shape_and_types(self) -> None:
        decomp = IVGNMD(max_iter=80)
        modes_t, modes_f, types, ridges, feats, f, t, i_spec, sk = decomp.fit_transform(
            self.signal, self.fs, return_all=True
        )
        k = modes_t.shape[0]
        self.assertGreaterEqual(k, 1)
        self.assertEqual(modes_t.shape[1], self.signal.size)
        self.assertEqual(modes_f.shape, (k, f.size))
        self.assertEqual(types.size, k)
        self.assertTrue(np.all(np.isin(types, [1, 2])))
        self.assertEqual(len(ridges), k)
        self.assertEqual(len(feats), k)
        self.assertEqual(i_spec.shape[1], self.signal.size)
        self.assertIsNotNone(decomp.modes_time_)

    def test_recovers_chirp_modes(self) -> None:
        decomp = IVGNMD(max_iter=120)
        modes = decomp.fit_transform(self.signal, self.fs)
        self.assertGreaterEqual(modes.shape[0], 2)
        # at least one chirp-like recovery against each true chirp
        for i in (0, 1):
            best = max(
                abs(float(np.corrcoef(self.true[i], modes[k])[0, 1]))
                for k in range(modes.shape[0])
            )
            self.assertGreater(best, 0.25)

    def test_functional_interface(self) -> None:
        modes = ivgnmd(self.signal, self.fs, max_iter=60)
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_default_call(self) -> None:
        decomp = IVGNMD(max_iter=40)
        a = decomp(self.signal, self.fs)
        b = decomp.fit_transform(self.signal, self.fs)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_str(self) -> None:
        self.assertIn("IVGNMD", str(IVGNMD()))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            IVGNMD().fit_transform(np.ones(10), self.fs)


if __name__ == "__main__":
    unittest.main()
