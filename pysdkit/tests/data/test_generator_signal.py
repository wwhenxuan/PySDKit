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
    """Test modules for verifying various basic signal generation algorithms"""

    duration = 1.0
    sample_rate = 1000
    noise_level = 0.0

    def test_add_noise(self) -> None:
        """Verify the function of adding noise"""
        for N in [10, 100, 500, 1000]:
            # Traversing different signal lengths
            for mean in [0.0, 0.1, -0.1, 0.5, -0.5, 5, 10, 20, 100, -100]:
                # Iterate over different means
                for std in [0.1, 0.5, 1, 2, 3, 5]:
                    # Iterate through different standard deviations
                    y = add_noise(N, mean, std)
                    self.assertTrue(expr=len(y) == N, msg="Output length error")

                    # Validate the numerical difference in noise means
                    diff_mean = np.allclose(np.mean(y), mean, atol=1e-6)
                    self.assertTrue(expr=diff_mean, msg="The mean value of the output noise does not meet the requirements")

                    # Verify the numerical difference in noise standard deviation
                    diff_std = np.allclose(np.std(y), std, atol=1e-6)
                    self.assertTrue(expr=diff_std, msg="The standard deviation of the output noise does not meet the requirements")

    def test_generate(self) -> None:
        """Test all functions that generate signals"""
        for fun in [
            generate_cos_signal,
            generate_sin_signal,
            generate_square_wave,
            generate_triangle_wave,
            generate_sawtooth_wave,
            generate_am_signal,
            generate_exponential_signal,
        ]:
            # Generate various signals
            time, signal = fun(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
            )
            # Determine whether the signal length and timestamp array length meet the requirements
            self.assertTrue(
                expr=len(signal) == self.sample_rate, msg="The length of the signal is wrong"
            )
            self.assertTrue(
                expr=len(time) == self.sample_rate, msg="Time error length error"
            )

            # Determine whether the timestamp array and signal length match
            self.assertTrue(
                expr=len(signal) == len(time), msg="The length of the signal and the length of the timestamp do not match"
            )

    def test_sine_generator(self) -> None:
        """Test sinusoidal signal generation"""
        for frequency in [1, 5, 10, 20]:
            # Iterate over different time frequencies to generate signals
            time, signal = generate_sin_signal(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )

            # Generates the specified signal
            sine = np.sin(2 * np.pi * frequency * time)

            # Determine the difference between the generated signal and the real signal
            diff = np.allclose(sine, signal, atol=1e-6)

            self.assertTrue(expr=diff, msg="The generated sinusoidal signal has numerical errors")

    def test_cosine_generator(self) -> None:
        """Test cosine signal generation"""
        for frequency in [1, 5, 10, 20]:
            # Iterate over different time frequencies to generate signals
            time, signal = generate_cos_signal(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )
            # Generates the specified cosine signal
            cosine = np.cos(2 * np.pi * frequency * time)
            # Determine the difference between the generated signal and the real signal
            diff = np.allclose(cosine, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="The generated cosine signal has numerical errors")

    def test_square_wave(self) -> None:
        """Verify square wave signal generation"""
        for frequency in [1, 5, 10, 20]:
            # Iterate over different time frequencies to generate signals
            time, signal = generate_square_wave(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )

            # Generates the specified square wave signal
            square = np.sign(np.sin(2 * np.pi * frequency * time))

            # Determine the difference between the generated signal and the real signal
            diff = np.allclose(square, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="The generated square wave signal has numerical errors")

    def test_triangle_wave(self) -> None:
        """Verify the generation of triangle wave signal"""
        for frequency in [1, 5, 10, 20]:
            # Iterate over different time frequencies to generate signals
            time, signal = generate_triangle_wave(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )

            # Generates the specified signal
            triangle = 2 * np.abs(sawtooth(2 * np.pi * frequency * time, 0.5)) - 1

            # Determine the difference between the generated signal and the real signal
            diff = np.allclose(triangle, signal, atol=1e-6)

            self.assertTrue(expr=diff, msg="The generated triangle wave signal has numerical errors")

    def test_sawtooth_wave(self) -> None:
        """Verify sawtooth signal generation"""
        for frequency in [1, 5, 10, 20]:
            # Iterate over different time frequencies to generate signals
            time, signal = generate_sawtooth_wave(
                duration=self.duration,
                sampling_rate=self.sample_rate,
                noise_level=self.noise_level,
                frequency=frequency,
            )

            # Generates the specified signal
            sawtooth_wave = sawtooth(2 * np.pi * frequency * time)

            # Determine the difference between the generated signal and the real signal
            diff = np.allclose(sawtooth_wave, signal, atol=1e-6)
            self.assertTrue(expr=diff, msg="The generated sawtooth signal has numerical errors")

    def test_exponential_signal(self) -> None:
        """Generation of validation index signals"""
        for decay_rate in [1.0, 2.0, -1.0, -2.0]:
            # Iterate through exponential decay factors of different sizes
            for initial_amplitude in [1.0, 2.0, -1.0, -2.0]:
                # Iterate over constant factors of different sizes
                time, signal = generate_exponential_signal(
                    duration=self.duration,
                    sampling_rate=self.sample_rate,
                    decay_rate=decay_rate,
                    initial_amplitude=initial_amplitude,
                    noise_level=self.noise_level,
                )

                # Generates the specified signal
                exp = initial_amplitude * np.exp(-decay_rate * time)

                # Determine the difference between the generated signal and the real signal
                diff = np.allclose(exp, signal, atol=1e-6)
                self.assertTrue(expr=diff, msg="The generated exponential signal has a numerical error")

    def test_am_signal(self) -> None:
        """Verify Amplitude Modulated (AM) signal generation"""
        for carrier_freq in [100, 120, 130, 150]:
            for modulating_freq in [5, 10, 20]:
                for mod_index in [1, 2]:
                    # Iterate over different time frequencies to generate signals
                    time, signal = generate_am_signal(
                        duration=self.duration,
                        sampling_rate=self.sample_rate,
                        noise_level=self.noise_level,
                        carrier_freq=carrier_freq,
                        modulating_freq=modulating_freq,
                        mod_index=mod_index,
                    )

                    # Carrier signal
                    carrier = np.cos(2 * np.pi * carrier_freq * time)

                    # Modulating signal
                    modulating_signal = np.cos(2 * np.pi * modulating_freq * time)

                    # Amplitude Modulated signal
                    am_signal = (1 + mod_index * modulating_signal) * carrier

                    # Determine the difference between the generated signal and the real signal
                    diff = np.allclose(am_signal, signal, atol=1e-6)
                    self.assertTrue(
                        expr=diff, msg="The generated Amplitude Modulated (AM) signal has numerical errors"
                    )


if __name__ == "__main__":
    unittest.main()
