# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pysdkit.tsa._stl import STL, STLResult
from pysdkit.data._time_series import generate_time_series

class TestSTL(unittest.TestCase):
    """测试STL分解算法能否正常运行"""

    def setUp(self) -> None:
        """创建测试用的时间序列数据"""
        np.random.seed(42)
        # 生成基础时间序列
        self.period = 12  # 季节性周期
        self.data = generate_time_series(
            duration=120,
            periodicities=np.array([self.period]),
            num_harmonics=np.array([2]),
            std=np.array([0.5])
        )
        # 添加线性趋势
        self.trend = np.linspace(0, 10, len(self.data))
        self.data += self.trend

    def test_fit_transform(self) -> None:
        """验证能否正常进行时间序列分解"""
        # 创建算法实例对象
        stl = STL(period=self.period)
        result = stl.fit_transform(self.data)
        
        # 验证返回结果类型
        self.assertIsInstance(result, STLResult, "返回结果类型错误")
        
        # 验证组件关系：observed = seasonal + trend + resid
        reconstructed = result.seasonal + result.trend + result.resid
        self.assertTrue(np.allclose(result.observed, reconstructed, atol=1e-10),
                        "分解后重建序列不匹配")

    def test_default_call(self) -> None:
        """验证call方法能否正常运行"""
        # 创建算法实例对象
        stl = STL(period=self.period)
        result = stl(self.data)
        
        # 验证基本输出
        self.assertIsInstance(result, STLResult, "返回结果类型错误")
        self.assertEqual(len(result.observed), len(self.data), "数据长度不一致")

    def test_robust_mode(self) -> None:
        """验证稳健模式下的分解"""
        # 创建包含异常值的数据
        data_with_outliers = self.data.copy()
        outlier_indices = [10, 30, 50, 70, 90]
        data_with_outliers[outlier_indices] += 10.0  # 添加异常值
        
        # 非稳健模式
        stl_normal = STL(period=self.period, robust=False)
        result_normal = stl_normal(data_with_outliers)
        
        # 稳健模式
        stl_robust = STL(period=self.period, robust=True)
        result_robust = stl_robust(data_with_outliers)
        
        # 比较残差：稳健模式在异常点应有更小的残差
        resid_normal = np.abs(result_normal.resid[outlier_indices])
        resid_robust = np.abs(result_robust.resid[outlier_indices])
        self.assertTrue(np.all(resid_robust < resid_normal), 
                        "稳健模式未正确处理异常值")

    def test_seasonal_component(self) -> None:
        """验证季节性分量的周期性"""
        stl = STL(period=self.period, seasonal=7)
        result = stl(self.data)
        
        # 验证季节性分量周期性
        seasonal = result.seasonal
        autocorr = np.correlate(seasonal, seasonal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 峰值应出现在周期点
        self.assertGreater(autocorr[0], autocorr[1], 
                          "季节性分量自相关峰值错误")
        self.assertGreater(autocorr[self.period], autocorr[self.period-1],
                          "季节性分量周期特性不明显")

    def test_trend_component(self) -> None:
        """验证趋势分量的平滑性"""
        stl = STL(period=self.period, trend=25)
        result = stl(self.data)
        
        # 计算趋势分量的一阶差分
        trend_diff = np.diff(result.trend)
        
        # 趋势分量应比原始数据平滑
        data_diff = np.diff(self.data)
        self.assertLess(np.std(trend_diff), np.std(data_diff),
                       "趋势分量不够平滑")
        
        # 趋势分量应与添加的线性趋势相关
        corr = np.corrcoef(result.trend, self.trend)[0, 1]
        self.assertGreater(corr, 0.9, "趋势分量与预期趋势相关性不足")

    def test_parameter_validation(self) -> None:
        """验证参数校验逻辑"""
        # 无效周期
        with self.assertRaises(ValueError):
            STL(period=1)
        
        # 无效seasonal参数
        with self.assertRaises(ValueError):
            STL(period=12, seasonal=4)  # 需要>=7且为奇数
        
        # 数据长度不足
        stl = STL(period=12)
        with self.assertRaises(ValueError):
            stl.fit_transform(np.random.rand(10))  # 10 < 2*12

    def test_different_iterations(self) -> None:
        """验证不同迭代次数的影响"""
        # 基础迭代
        stl_base = STL(period=self.period)
        result_base = stl_base(self.data)
        
        # 增加内迭代次数
        stl_more_inner = STL(period=self.period)
        result_inner = stl_more_inner.fit_transform(self.data, inner_iter=5)
        
        # 增加外迭代次数（稳健模式）
        stl_robust = STL(period=self.period, robust=True)
        result_robust = stl_robust.fit_transform(self.data, outer_iter=15)
        
        # 比较残差标准差
        std_base = np.std(result_base.resid)
        std_inner = np.std(result_inner.resid)
        std_robust = np.std(result_robust.resid)
        
        # 更多迭代应产生更小的残差
        self.assertLess(std_inner, std_base * 1.05,
                       "增加内迭代未改善分解效果")
        self.assertLess(std_robust, std_base * 1.05,
                       "增加外迭代未改善分解效果")

if __name__ == "__main__":
    unittest.main()