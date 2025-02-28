import unittest

from pysdkit.data import test_emd

from pysdkit import FAEMD


class FAEMDTest(unittest.TestCase):
    """测试快速自适应经验模态分解算法能否正常运行"""

    def test_fit_transform(self) -> None:
        """验证能否正常进行信号分解"""
        # 创建算法验证的实例
        faemd = FAEMD(max_imfs=3)

        # 获取一元测试信号
        time, signal = test_emd()

        # 测试一元信号
        for num_imfs in range(2, 5):
            # 执行信号分解算法
            IMFs = faemd.fit_transform(signal, max_imfs=num_imfs)
            # 获取分解得到的本征模态函数
            num_vars, seq_len = IMFs.shape


if __name__ == "__main__":
    unittest.main()
