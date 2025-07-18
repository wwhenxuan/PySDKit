# -*- coding: utf-8 -*-
"""
Created on 2025/02/16 00:11:14
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np
from scipy.signal import sawtooth

from pysdkit.data import (
    add_noise,
    generate_cos_signal,
    generate_sin_signal,
    generate_square_wave,
    generate_triangle_wave,
    generate_sawtooth_wave,
    generate_am_signal,
    generate_exponential_signal,
)


class Base_Signal_Generator_Test(unittest.TestCase):
    """验证各种生成基础形式信号生成算法的测试模块"""

    duration = 1.0
    sample_rate = 1000
    noise_level = 0.0

    def test_add_noise(self) -> None:
        """验证添加噪声的函数"""
        for N in [10, 100, 500, 1000]:
            # 遍历不同的信号长度
            for mean in [0.0, 0.1, -0.1, 0.5, -0.5, 5, 10, 20, 100, -100]:
                # 遍历不同的均值
                for std in [0.1, 0.5, 1, 2, 3, 5]:
                    # 遍历不同的标准差
                    y = add_noise(N, mean, std)
                    self.assertTrue(expr=len(y) == N, msg="输出长度错误")
                    # 验证噪声均值的数值差异
                    diff_mean = np.allclose(np.mean(y), mean, atol=1e-6)
                    self.assertTrue(expr=diff_mean, msg="输出噪声的均值不符合要求")
                    # 验证噪声标准差的数值差异
                    diff_std = np.allclose(np.std(y), std, atol=1e-6)
                    self.assertTrue(expr=diff_std, msg="输出噪声的标准差不符合要求")

    def test_generate(self) -> None:
        """测试所有生成信号的函数"""
        for fun in [
            generate_cos_signal,
            generate_sin_signal,
            generate_square_wave,
            generate_triangle_wave,
            generate_sawtooth_wave,
            generate_am_signal,
            generate_exponential_signal,
        ]:
            # 生成各种形式的信号
            time, signal = fun(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
            )
            # 判断信号长度和时间戳数组的长度是否符合要求
            self.assertTrue(
                expr=len(signal) == self.sample_rate, msg="信号的长度出现错误"
            )
            self.assertTrue(
                expr=len(time) == self.sample_rate, msg="时间错长度出现错误"
            )
            # 判断时间戳数组和信号长度是否匹配
            self.assertTrue(
                expr=len(signal) == len(time), msg="信号的长度和时间戳的长度不匹配"
            )

    def test_sine_generator(self) -> None:
        """测试正弦信号的生成"""
        for frequency in [1, 5, 10, 20]:
            # 遍历不同的时间频率生成信号
            time, signal = generate_sin_signal(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )
            # 生成指定的信号
            sine = np.sin(2 * np.pi * frequency * time)
            # 判断生成信号和真实信号的差异
            diff = np.allclose(sine, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="生成的正弦信号存在数值错误")

    def test_cosine_generator(self) -> None:
        """测试余弦信号的生成"""
        for frequency in [1, 5, 10, 20]:
            # 遍历不同的时间频率生成信号
            time, signal = generate_cos_signal(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )
            # 生成指定的余弦信号
            cosine = np.cos(2 * np.pi * frequency * time)
            # 判断生成信号和真实信号的差异
            diff = np.allclose(cosine, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="生成的余弦信号存在数值错误")

    def test_square_wave(self) -> None:
        """验证方波信号的生成"""
        for frequency in [1, 5, 10, 20]:
            # 遍历不同的时间频率生成信号
            time, signal = generate_square_wave(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )
            # 生成指定的方波信号
            square = np.sign(np.sin(2 * np.pi * frequency * time))
            # 判断生成信号和真实信号的差异
            diff = np.allclose(square, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="生成的方波信号存在数值错误")

    def test_triangle_wave(self) -> None:
        """验证三角波信号的生成"""
        for frequency in [1, 5, 10, 20]:
            # 遍历不同的时间频率生成信号
            time, signal = generate_triangle_wave(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )
            # 生成指定的信号
            triangle = 2 * np.abs(sawtooth(2 * np.pi * frequency * time, 0.5)) - 1
            # 判断生成信号和真实信号的差异
            diff = np.allclose(triangle, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="生成的三角波信号存在数值错误")

    def test_sawtooth_wave(self) -> None:
        """验证sawtooth信号的生成"""
        for frequency in [1, 5, 10, 20]:
            # 遍历不同的时间频率生成信号
            time, signal = generate_sawtooth_wave(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )
            # 生成指定的信号
            sawtooth_wave = sawtooth(2 * np.pi * frequency * time)
            # 判断生成信号和真实信号的差异
            diff = np.allclose(sawtooth_wave, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="生成的sawtooth信号存在数值错误")

    def test_exponential_signal(self) -> None:
        """验证指数信号的生成"""
        for decay_rate in [1.0, 2.0, -1.0, -2.0]:
            # 遍历不同大小的指数衰减因子
            for initial_amplitude in [1.0, 2.0, -1.0, -2.0]:
                # 遍历不同大小的常数因子
                time, signal = generate_exponential_signal(
                    duration=self.duration,
                    sampling_rate=self.sample_rate,
                    decay_rate=decay_rate,
                    initial_amplitude=initial_amplitude,
                    noise_level=self.noise_level,
                )
                # 生成指定的信号
                exp = initial_amplitude * np.exp(-decay_rate * time)
                # 判断生成信号和真实信号的差异
                diff = np.allclose(exp, signal, atol=1e-6)
                self.assertTrue(expr=diff, msg="生成的指数信号存在数值错误")

    def test_am_signal(self) -> None:
        """验证Amplitude Modulated (AM)信号的生成"""
        for carrier_freq in [100, 120, 130, 150]:
            for modulating_freq in [5, 10, 20]:
                for mod_index in [1, 2]:
                    # 遍历不同的时间频率生成信号
                    time, signal = generate_am_signal(
                        duration=self.duration,
                        sampling_rate=self.sample_rate,
                        noise_level=self.noise_level,
                        carrier_freq=carrier_freq,
                        modulating_freq=modulating_freq,
                        mod_index=mod_index,
                    )
                    # 生成指定的信号
                    # Carrier signal
                    carrier = np.cos(2 * np.pi * carrier_freq * time)
                    # Modulating signal
                    modulating_signal = np.cos(2 * np.pi * modulating_freq * time)
                    # Amplitude Modulated signal
                    am_signal = (1 + mod_index * modulating_signal) * carrier
                    # 判断生成信号和真实信号的差异
                    diff = np.allclose(am_signal, signal, atol=1e-6)
                    self.assertTrue(
                        expr=diff, msg="生成的Amplitude Modulated (AM)信号存在数值错误"
                    )


if __name__ == "__main__":
    unittest.main()
