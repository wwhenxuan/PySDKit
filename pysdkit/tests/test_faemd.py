import unittest

import matplotlib.pyplot as plt
import numpy as np

from pysdkit.data import test_emd, test_multivariate_signal

from pysdkit import FAEMD


class FAEMDTest(unittest.TestCase):
    """测试快速自适应经验模态分解算法能否正常运行"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""
        # 创建算法验证的实例
        faemd = FAEMD(max_imfs=3)

        # 获取一元测试信号
        _, signal = test_emd()

        # 测试一元信号
        for num_imfs in range(2, 5):
            # 执行信号分解算法
            IMFs = faemd.fit_transform(signal, max_imfs=num_imfs)

            # 获取分解得到的本征模态函数
            num_vars, seq_len = IMFs.shape

            # 检查信号的分解模态数目
            self.assertEqual(first=num_vars, second=num_imfs, msg=f"分解模态的数目错误")

            # 检查信号的长度
            self.assertEqual(first=seq_len, second=len(signal), msg="分解信号的长度错误")

            # 检验信号分解效果
            diff = np.allclose(np.sum(IMFs, axis=0), signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="分解子模态无法重构原信号")

        # 获取二元测试信号
        _, signal = test_multivariate_signal()
        # 获取信号的形状信息
        num_vars, seq_len = signal.shape

        # 执行多元信号分解算法
        IMFs = faemd.fit_transform(signal, max_imfs=num_vars)

        # 检验分解子模态的数目
        self.assertEqual(first=IMFs.shape[0], second=num_vars, msg="分解信号的模态数目错误")

        # 检验分解信号的长度
        self.assertEqual(first=IMFs.shape[1], second=seq_len, msg="分解信号的长度错误")

        # 检验信号的重构效果
        for num in range(num_vars):
            diff = np.allclose(np.sum(IMFs[:, :, num], axis=0), signal[num], atol=1e-6)
            self.assertTrue(expr=diff, msg="分解子模态无法重构原信号")

    def test_default_call(self) -> None:
        """验证能否正常进行信号分解"""
        # 创建算法验证的实例
        faemd = FAEMD(max_imfs=3)

        # 获取一元测试信号
        _, signal = test_emd()

        # 测试一元信号
        for num_imfs in range(2, 5):
            # 执行信号分解算法
            IMFs = faemd(signal, max_imfs=num_imfs)

            # 获取分解得到的本征模态函数
            num_vars, seq_len = IMFs.shape

            # 检查信号的分解模态数目
            self.assertEqual(first=num_vars, second=num_imfs, msg=f"分解模态的数目错误")

            # 检查信号的长度
            self.assertEqual(first=seq_len, second=len(signal), msg="分解信号的长度错误")

            # 检验信号分解效果
            diff = np.allclose(np.sum(IMFs, axis=0), signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="分解子模态无法重构原信号")

        # 获取二元测试信号
        _, signal = test_multivariate_signal()
        # 获取信号的形状信息
        num_vars, seq_len = signal.shape

        # 执行多元信号分解算法
        IMFs = faemd(signal, max_imfs=num_vars)

        # 检验分解子模态的数目
        self.assertEqual(first=IMFs.shape[0], second=num_vars, msg="分解信号的模态数目错误")

        # 检验分解信号的长度
        self.assertEqual(first=IMFs.shape[1], second=seq_len, msg="分解信号的长度错误")

        # 检验信号的重构效果
        for num in range(num_vars):
            diff = np.allclose(np.sum(IMFs[:, :, num], axis=0), signal[num], atol=1e-6)
            self.assertTrue(expr=diff, msg="分解子模态无法重构原信号")

    def test_trend(self) -> None:
        """判断对于单一的趋势信号输入"""
        faemd = FAEMD(max_imfs=3)

        # 创建仅有趋势分量的时间戳和信号
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # 执行信号分解算法获得本征模态函数
        IMFs = faemd.fit_transform(signal=signal)

        # 检验信号的最后一个本征模态函数
        diff = np.allclose(IMFs[-1], signal, atol=1e-6)
        self.assertTrue(expr=diff, msg="没能正确提取趋势信息")

    def test_signal_imf(self) -> None:
        """验证算法对单一模态信号的提取能力"""
        # 创建信号分解算法实例
        faemd = FAEMD(max_imfs=2, tol=1e-10)

        # 创建时间戳数组
        time = np.arange(0, 1, 0.001)

        # 创建余弦信号
        cosine = np.cos(2 * np.pi * 4 * time)

        # 判断单一余弦函数的输入
        IMFs = faemd.fit_transform(signal=cosine.copy())

        # 检验信号的最后一个本征模态函数
        diff = np.allclose(IMFs[0], cosine, atol=1)
        self.assertTrue(expr=diff, msg="没能正确提取单一模态信息")

    def test_window_type(self) -> None:
        """验证使用的平滑算法类型"""
        # 创建测试信号
        time, signal = test_emd()

        # 遍历指定的索引类型
        for index in range(7):
            # 创建测试信号实例
            faemd = FAEMD(max_imfs=2, window_type=index)

            # 执行信号分解算法
            IMFs = faemd.fit_transform(signal=signal)

            # 判断差异
            diff = np.allclose(np.sum(IMFs, axis=0), signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="分解得到的本征模态函数无法重构原信号")

    def test_wrong_window_type(self) -> None:
        """验证错误的`window_type`参数"""
        with self.assertRaises(ValueError):
            # 以错误的参数创建信号分解算法实例
            FAEMD(max_imfs=2, window_type=-1)


if __name__ == "__main__":
    unittest.main()
