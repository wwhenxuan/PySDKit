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

    def test_wrong_shape_inputs(self) -> None:
        """测试K近邻分类器的非法的形状输入"""
        # 创建算法实例
        knn = KnnDtw(n_neighbors=1)

        # 测试非法的输入样本，少了一个维度
        with self.assertRaises(ValueError):
            knn.fit(samples=np.ones(100), labels=np.ones(100))

        # 测试非法的输入样本，多了一个维度
        with self.assertRaises(ValueError):
            knn.fit(samples=np.ones(shape=(100, 100, 100)), labels=np.ones(100))

        # 测试非法的输入标签，多了一个维度
        with self.assertRaises(ValueError):
            knn.fit(samples=np.ones(shape=(100, 100)), labels=np.ones(shape=(100, 100)))

        # 测试输入的样本数目和标签数目不相等
        with self.assertRaises(ValueError):
            knn.fit(samples=np.ones(shape=(100, 50)), labels=np.ones(99))

    def test_wrong_type_inputs(self) -> None:
        """测试K近邻分类器的非法数据类型输入"""
        # 创建算法实例
        knn = KnnDtw(n_neighbors=1)

        # 测试非法的样本输入
        with self.assertRaises(TypeError):
            knn.fit(1, np.ones(100))

        # 测试非法的标签输入
        with self.assertRaises(TypeError):
            knn.fit(np.ones(shape=(10, 100)), 1)

    def test_fit(self) -> None:
        """测试该模型中的`fit`方法能否正确执行"""
        # 生成用于测试的随机数据
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # 创建K近邻分类器实例对象
        knn = KnnDtw(n_neighbors=1)

        # 拟合数据
        knn.fit(X, y)

        # 判断拟合的数据
        self.assertEqual(first=X.all(), second=knn.samples.all(), msg="输入的样本与拟合样本不一致")
        self.assertEqual(first=y.all(), second=knn.labels.all(), msg="输入的标签与拟合的标签不一致")

    def test_predict(self) -> None:
        """测试K近邻分类器中用于预测的方法"""
        # 生成用于测试的随机数据
        X = self.rng.random(size=(10, 2))
        y = np.hstack((np.zeros(5), np.ones(5)))

        # 创建K近邻分类器实例对象
        knn = KnnDtw(n_neighbors=1)

        # 拟合数据
        knn.fit(X, y)
        # 预测样本
        (y_pred, y_prob) = knn.predict(X)

        # 判断前后标签是否一致
        self.assertEqual(first=y.all(), second=y_pred.all(), msg="最近邻算法分类错误")

        # 判断标签和概率的长度
        self.assertEqual(first=len(y), second=len(y_pred), msg="输出标签长度错误")
        self.assertEqual(first=len(y), second=len(y_prob), msg="输出概率标签长度错误")


if __name__ == '__main__':
    unittest.main()
