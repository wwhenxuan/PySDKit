# -*- coding: utf-8 -*-
"""
Created on 2025/07/22 15:21:32
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit.models import PCA
from pysdkit.data import test_pca


class PCATest(unittest.TestCase):
    """测试PCA主成分分析算法能否正常运行"""

    def test_create_pca(self) -> None:
        """验证主成分分析算法能否正确创建"""
        for n_components in range(2, 6):
            # 实例化主成分分析的对象
            pca = PCA(n_components=n_components)
            # 判断创建的实例是否与目标实例一致
            self.assertIsInstance(pca, PCA, msg="创建的PCA对象类别错误")
            # 判断类中的属性参数是否与传入的超参数一致
            self.assertEqual(
                pca.n_components,
                n_components,
                msg="创建的PCA对象中的属性参数与传入的超参数不一致",
            )
    def test_none(self) -> None:
        """测试在未执行`fit_transform`方法前PCA中的初始化数据属性是否为None"""
        # 创建K近邻分类器实例对象
        pca = PCA(n_components=2)
        # 检验初始的拟合数据是否为None
        self.assertIsNone(pca.X_reduced, msg="创建的PCA对象在未拟合数据时数据对象为非None")
        self.assertIsNone(pca._components, msg="创建的PCA对象在未拟合数据时数据对象为非None")
        self.assertIsNone(pca._explained_variance_ratio, msg="创建的PCA对象在未拟合数据时数据对象为非None")

    def test_wrong_shape_inputs(self) -> None:
        """测试PCA算法的非法的形状输入"""
        # 创建PCA算法实例对象
        pca = PCA(n_components=2)

        # 测试非法的输入样本，少了一个维度
        with self.assertRaises(ValueError):
            pca.fit_transform(X=np.ones(100))

        # 测试非法的输入样本，多了一个维度
        with self.assertRaises(ValueError):
            pca.fit_transform(X=np.ones(shape=(10, 10, 100)))

    def test_wrong_type_inputs(self) -> None:
        """测测试非法的数据类型输入"""
        # 创建PCA算法实例对象
        pca = PCA(n_components=2)

        with self.assertRaises(TypeError):
            # 测试非法的数据类型输入
            pca.fit_transform(X=42)

    def test_fit_transform(self) -> None:
        """测试PCA算法能否正确执行"""
        X = ...