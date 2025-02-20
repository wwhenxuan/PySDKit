# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 16:18:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import sys
import unittest

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # 创建测试用例加载器
    test_suite = unittest.defaultTestLoader.discover(".", "*test_*.py")
    # 测试用例运行器
    test_runner = unittest.TextTestRunner(
        resultclass=unittest.TextTestResult, verbosity=2
    )
    # 执行当前目录下所有的测试用例
    result = test_runner.run(test_suite)  # run方法将返回测试结果
    sys.exit(not result.wasSuccessful())
