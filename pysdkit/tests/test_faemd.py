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
            # 检查信号的分解模态数目
            self.assertEqual(first=num_vars, second=num_imfs, msg=f"分解模态的数目错误")
            # 检查信号的长度
            self.assertEqual(first=seq_len, second=len(signal), msg="分解信号的长度错误")












if __name__ == "__main__":
    unittest.main()
