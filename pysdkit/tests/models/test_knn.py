# -*- coding: utf-8 -*-
"""
Created on 2025/07/22 10:08:57
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit.models import KnnDtw


class KNNTest(unittest.TestCase):
    """测试K近邻分类器能否正常运行"""

    # 创建用于测试的随机数生成器
    rng = np.random.RandomState(42)

    def test_create_knn(self) -> None:
        """测试K近邻分类器能否正确正确创建"""
        for n_neighbors in range(1, 6):
            # 创建类属性的实例对象
            knn = KnnDtw(n_neighbors=n_neighbors)
            # 判断创建的实例是否与目标实例一致
            self.assertIsInstance(knn, KnnDtw, msg="创建的KNN对象类别错误")
            # 判断类中的属性参数是否与传入的超参数一致
            self.assertEqual(knn.n_neighbors, n_neighbors, msg="创建的KNN对象中的属性参数与传入的超参数不一致")

    def test_none(self) -> None:
        """测试在未执行`fit`方法前K近邻分类器中的初始化数据属性是否为None"""
        # 创建K近邻分类器实例对象
        knn = KnnDtw(n_neighbors=1)
        # 检验初始的拟合数据是否为None
        self.assertIsNone(knn.samples, msg="创建的KNN对象在未拟合数据时数据对象为非None")
        self.assertIsNone(knn.labels, msg="创建的KNN对象在未拟合数据时数据对象为非None")

    def test_fit(self) -> None:
        """测试该模型中的`fit`方法能否正确执行"""
        # 生成用于测试的随机数据
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # 创建K近邻分类器实例对象
        knn = KnnDtw(n_neighbors=1)
        # 检验初始的拟合数据是否为None

    def test_wrong_inputs(self) -> None:
        """测试K近邻分类器的非法输入"""







if __name__ == '__main__':
    unittest.main()