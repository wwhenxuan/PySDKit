# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 16:18:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import EMD
from pysdkit.data import test_emd, test_univariate_signal


class EMDTest(unittest.TestCase):
    """测试经验模态分解算法(EMD)能否正常运行"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""
        # 创建算法实例对象
        emd = EMD()
        for index in range(1, 4):
            # 获取测试信号
            time, signal = test_univariate_signal(case=index)
            IMFs = emd.fit_transform(signal)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            _, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")

    def test_default_call(self) -> None:
        """验证call方法能否正常运行"""
        time, signal = test_emd()
        # 创建算法实例对象
        emd = EMD()
        IMFs = emd(signal)

        # 判断输出的维数
        dim = len(IMFs.shape)
        self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
        # 判断输出信号的长度
        _, length = IMFs.shape
        self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")

    def test_different_length_inputs(self) -> None:
        """验证当时间戳数组和输入信号长度不一致时的异常"""
        time = np.arange(100)
        signal = np.random.randn(125)

        emd = EMD()
        with self.assertRaises(ValueError):
            emd.fit_transform(signal=signal, time=time)

    def test_trend(self) -> None:
        """判断对于单一的趋势信号输入"""
        emd = EMD()

        # 创建仅有趋势分量的时间戳和信号
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # 执行信号分解算法获得本征模态函数
        IMFs = emd.fit_transform(signal=signal, time=time)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF")
        self.assertTrue(np.allclose(signal, IMFs[0]))

    def test_single_imf(self) -> None:
        """判断单一本征模态函数的输入"""
        emd = EMD()

        # 创建时间戳数组
        time = np.arange(0, 1, 0.001)

        # 创建余弦信号
        cosine = np.cos(2 * np.pi * 4 * time)

        # 判断单一余弦函数的输入
        IMFs = emd.fit_transform(signal=cosine.copy(), time=time)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF!")

        # 判断输入输出之间的数值差异
        diff = np.allclose(IMFs[0], cosine)
        self.assertTrue(diff, "Expecting 1st IMF to be cos(8 * pi * t)")

        # 创建输入的趋势分量
        trend = 3 * (time - 0.5)

        # 判断余弦与趋势分量输入
        IMFs = emd.fit_transform(signal=trend.copy() + cosine.copy(), time=time)
        self.assertEqual(
            first=IMFs.shape[0], second=2, msg="Expecting two IMF of cosine and trend!"
        )

        # 进一步判断两个模态输出的数值差异
        diff_cosine = np.allclose(IMFs[0], cosine, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 1st IMF to be cosine")
        diff_trend = np.allclose(IMFs[1], trend, atol=0.2)
        self.assertTrue(diff_trend, "Expecting 2nd IMF to be trend")

    def test_spline_kind(self) -> None:
        """验证EMD算法中所有的插值算法能够正常运行"""
        # 创建两个不同分量的测试信号
        time = np.arange(0, 1, 0.01)
        cosine = np.cos(2 * np.pi * 4 * time)
        trend = 3 * (time - 0.1)
        signal = cosine.copy() + trend.copy()

        for spline_kind in [
            "akima",
            "cubic",
            "pchip",
            "cubic_hermite",
            "slinear",
            "quadratic",
            "linear",
        ]:
            # 遍历所有的插值算法并创建实例
            emd = EMD(spline_kind=spline_kind)

            # 执行信号分解算法并判断输出结果
            IMFs = emd.fit_transform(signal=signal, time=time)

            # 判断IMF的数目是否符合要求
            self.assertEqual(
                first=IMFs.shape[0],
                second=2,
                msg=f"Expecting two IMF of cosine and trend when the `spline_kind` is {spline_kind}",
            )

            # 进一步判断两个模态输出的数值差异
            diff_cosine = np.allclose(IMFs[0], cosine, atol=0.3)
            self.assertTrue(
                diff_cosine,
                msg=f"Expecting 1st IMF to be cosine when the `spline_kind` is {spline_kind}",
            )
            diff_trend = np.allclose(IMFs[1], trend, atol=0.3)
            self.assertTrue(
                diff_trend,
                msg=f"Expecting 2nd IMF to be trend when the `spline_kind` is {spline_kind}",
            )

    def test_wrong_spline_kind(self) -> None:
        """验证错误的插值类型输入是否会引发异常"""
        spline_kind = "wrong"

        # 创建随机输入
        time = np.arange(10)
        signal = np.random.randn(10)

        # 开始验证错误的插值类型
        with self.assertRaises(ValueError):
            emd = EMD(spline_kind=spline_kind)
            emd.fit_transform(signal=signal, time=time)

    def test_extrema_detection(self) -> None:
        """验证EMD算法中所有的极值点探测算法能否正常的运行"""
        # 创建两个不同分量的测试信号
        time = np.arange(0, 1, 0.01)
        cosine = np.cos(2 * np.pi * 4 * time)
        trend = 3 * (time - 0.1)
        signal = cosine.copy() + trend.copy()

        for extrema_detection in ["parabol", "simple"]:
            # 遍历所有的插值算法并创建实例
            emd = EMD(extrema_detection=extrema_detection)

            # 执行信号分解算法并判断输出结果
            IMFs = emd.fit_transform(signal=signal, time=time)

            # 判断IMF的数目是否符合要求
            self.assertEqual(
                first=IMFs.shape[0],
                second=2,
                msg=f"Expecting two IMF of cosine and trend when the `spline_kind` is {extrema_detection}",
            )

            # 进一步判断两个模态输出的数值差异
            diff_cosine = np.allclose(IMFs[0], cosine, atol=0.3)
            self.assertTrue(
                diff_cosine,
                msg=f"Expecting 1st IMF to be cosine when the `extrema_detection` is {extrema_detection}",
            )
            diff_trend = np.allclose(IMFs[1], trend, atol=0.3)
            self.assertTrue(
                diff_trend,
                msg=f"Expecting 2nd IMF to be trend when the `extrema_detection` is {extrema_detection}",
            )

    def test_wrong_extrema_detection(self) -> None:
        """验证错误的极值点探测类型输入是否会引发异常"""
        extrema_detection = "wrong"

        # 创建随机输入
        time = np.arange(10)
        signal = np.random.randn(10)

        # 开始验证错误的极值点探测类型
        with self.assertRaises(ValueError):
            emd = EMD(extrema_detection=extrema_detection)
            emd.fit_transform(signal=signal, time=time)

    def test_max_iteration_flag(self) -> None:
        """验证模型的最大迭代次数"""
        # 创建随机信号进行验证
        signal = np.random.random(200)
        emd = EMD()
        emd.MAX_ITERATION = 10
        emd.FIXE = 20
        IMFs = emd.fit_transform(signal)

        # There's not much to test, except that it doesn't fail.
        # With low MAX_ITERATION value for random signal it's
        # guaranteed to have at least 2 IMFs.
        self.assertTrue(IMFs.shape[0] > 1)

    def test_get_imfs_and_residue(self) -> None:
        """验证经过分解后能否正常获得本征模态函数和趋势分量"""
        signal = np.random.random(200)
        emd = EMD(**{"MAX_ITERATION": 10, "FIXE": 20})
        all_imfs = emd(signal, max_imfs=3)

        imfs, residue = emd.get_imfs_and_residue()
        self.assertEqual(
            all_imfs.shape[0], imfs.shape[0] + 1, msg="Compare number of components"
        )
        self.assertTrue(
            np.array_equal(all_imfs[:-1], imfs),
            msg="Shouldn't matter where imfs are from",
        )
        self.assertTrue(
            np.array_equal(all_imfs[-1], residue),
            msg="Residue, if any, is the last row",
        )

    def test_get_imfs_and_residue_without_running(self) -> None:
        """验证当不执行算法时能不能获得输出的结果"""
        emd = EMD()
        with self.assertRaises(ValueError):
            # 由于没有执行分解过程因此按理说无法获得IMFs和残差结果
            _, _ = emd.get_imfs_and_residue()

    def test_get_imfs_and_trend(self) -> None:
        """验证经过分解后能否正常获得本征模态函数和趋势分量"""
        # 创建算法实例和测试信号
        emd = EMD()
        time = np.linspace(0, 2 * np.pi, 100)
        expected_trend = 5 * time
        signal = (
            2 * np.sin(4.1 * 6.28 * time)
            + 1.2 * np.cos(7.4 * 6.28 * time)
            + expected_trend
        )

        # 执行信号分解算法
        IMFs = emd(signal)
        # 尝试获得趋势分量
        imfs, trend = emd.get_imfs_and_trend()

        # 对趋势分量进行进一步的数值验证
        onset_trend = trend - trend.mean()
        onset_expected_trend = expected_trend - expected_trend.mean()
        self.assertEqual(
            IMFs.shape[0], imfs.shape[0] + 1, "Compare number of components"
        )
        self.assertTrue(
            np.array_equal(IMFs[:-1], imfs), "Shouldn't matter where imfs are from"
        )
        self.assertTrue(
            np.allclose(onset_trend, onset_expected_trend, rtol=0.1, atol=0.5),
            "Extracted trend should be close to the actual trend",
        )


if __name__ == "__main__":
    unittest.main()
