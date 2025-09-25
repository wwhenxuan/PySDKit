# -*- coding: utf-8 -*-
"""
Created on 2025/02/21 23:40:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import ewt, EWT
from pysdkit.data import test_emd


class EWTTest(unittest.TestCase):
    """测试经验小波变换算法(EMD)能否正常运行"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""
        # 生成测试函数样例
        time, signal = test_emd()
        # 遍历不同的分解模态数目
        for K in range(2, 6):
            # 创建信号分解实例对象
            ewt_c = EWT(K=K)
            # 执行信号分解算法
            IMFs = ewt_c.fit_transform(signal)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            number, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")
            # 判断分解模态的数目
            self.assertEqual(first=number, second=K)

    def test_default_call(self) -> None:
        """验证call方法能够正常运行"""
        # 生成测试函数样例
        time, signal = test_emd()
        # 遍历不同的分解模态数目
        for K in range(2, 6):
            # 创建信号分解实例对象
            ewt_c = EWT(K=K)
            # 执行信号分解算法
            IMFs = ewt_c(signal)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            number, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")
            # 判断分解模态的数目
            self.assertEqual(first=number, second=K)

    def test_ewt_function(self) -> None:
        """测试经验小波变换的函数"""
        # 生成测试函数样例
        time, signal = test_emd()
        # 遍历不同的分解模态数目
        for K in range(2, 6):
            # 执行信号分解算法
            IMFs = ewt(signal, K=K)
            # 判断输出的维数
            dim = len(IMFs.shape)
            self.assertEqual(first=dim, second=2, msg="分解信号的输出形状错误")
            # 判断输出信号的长度
            number, length = IMFs.shape
            self.assertEqual(first=len(signal), second=length, msg="分解信号的长度错误")
            # 判断分解模态的数目
            self.assertEqual(first=number, second=K)

    def test_fmirror(self) -> None:
        """验证EWT算法自带的镜像拓展算法"""
        inputs = np.array([1, 2, 3, 4, 5])
        # 创建待测试的实例
        ewt_c = EWT()
        for sym in range(1, len(inputs)):
            outputs = ewt_c.fmirror(inputs, sym=sym, end=1)
            # 检验镜像拓展的长度与预期是否匹配
            self.assertEqual(
                first=len(outputs),
                second=len(inputs) + 2 * sym - 1,
                msg="镜像拓展函数的结果与预期不符",
            )

    def test_detect(self) -> None:
        """测试边界检测的各种方法"""
        # 生成测试信号
        time, signal = test_emd()

        # 遍历各种边界检测方法
        for detect in ["locmax", "locmaxmin", "locmaxminf"]:
            # 创建信号分解算法的实例
            ewt_c = EWT(K=5, detect=detect)
            # 执行信号分解算法
            IMFs = ewt_c(signal)
            # 检验是否能够正常分解
            num_imfs, length = IMFs.shape
            self.assertEqual(
                first=length, second=len(signal), msg="分解信号的长度与原信号不相等"
            )

    def test_wrong_detect(self) -> None:
        """测试错误的边界检测方法"""
        # 生成测试信号
        time, signal = test_emd()

        # 错误的方法
        detect = "wrong"

        # 使用错误的参数创建算法实例
        ewt_c = EWT(K=5, detect=detect)
        with self.assertRaises(ValueError):
            ewt_c(signal)

    def test_reg(self) -> None:
        """Test the regularization method applied to the filter bank"""
        # 生成测试信号
        time, signal = test_emd()

        # 遍历各种regularization method方法的列表
        for reg in ["average", "gaussian", "wrong"]:
            # 创建信号分解算法的实例
            ewt_c = EWT(K=3, reg=reg)
            # 执行信号分解算法
            IMFs = ewt_c(signal)
            # 检验是否能够正常分解
            num_imfs, length = IMFs.shape
            self.assertEqual(
                first=num_imfs, second=3, msg="分解信号的本征模态函数数目错误"
            )
            self.assertEqual(
                first=length, second=len(signal), msg="分解信号的长度与原信号不相等"
            )


if __name__ == "__main__":
    unittest.main()
