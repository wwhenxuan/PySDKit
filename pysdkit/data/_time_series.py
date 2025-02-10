# -*- coding: utf-8 -*-
"""
Created on 2025/02/10 18:35:29
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Generate time series test examples
"""
import numpy as np
from typing import Optional, Union


def simulate_seasonal_term(
    periodicity: Union[float, np.ndarray],
    total_cycles: Union[float, np.ndarray],
    noise_std: Union[float, np.ndarray] = 1.0,
    harmonics=None,
) -> np.ndarray:
    """
    This function is adapted from the famous statistical analysis library in Python, statsmodels.
    It synthesizes time series data based on a specified periodicity, number of cycles, and noise.

    Code reference: https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_seasonal.html

    :param periodicity: The length of one cycle in the seasonal pattern (e.g., the number of time steps per cycle).
    :param total_cycles: The total number of full cycles to generate in the time series data.
    :param noise_std: The standard deviation of the noise to be added to the seasonal components.
    :param harmonics: The number of harmonics (sinusoidal components) used to generate the seasonal pattern.
                       If None, it defaults to half the periodicity.
    :return: A numpy array representing the generated time series data with seasonal behavior.
    """

    # Calculate the total duration of the time series (in terms of time steps).
    duration = periodicity * total_cycles
    # Ensure that the total duration is an integer.
    assert duration == int(duration)
    duration = int(duration)

    # Set the number of harmonics if not provided. Default is half the periodicity.
    harmonics = harmonics if harmonics else int(np.floor(periodicity / 2))

    # Calculate the angular frequency corresponding to the periodicity.
    lambda_p = 2 * np.pi / float(periodicity)

    # Initialize the seasonal noise components for each harmonic (both cosine and sine components).
    gamma_jt = noise_std * np.random.randn(harmonics)
    gamma_star_jt = noise_std * np.random.randn(harmonics)

    # Define the total number of time steps, including extra time for burn-in.
    total_timesteps = 100 * duration  # Pad for burn-in phase
    # Initialize an array to hold the generated time series.
    series = np.zeros(total_timesteps)

    # Loop over each time step to generate the time series data.
    for t in range(total_timesteps):
        # Initialize new gamma values for the next time step.
        gamma_jtp1 = np.zeros_like(gamma_jt)
        gamma_star_jtp1 = np.zeros_like(gamma_star_jt)

        # For each harmonic, update its seasonal components (cosine and sine) based on the previous values.
        for j in range(1, harmonics + 1):
            # Calculate the cosine and sine values for this harmonic.
            cos_j = np.cos(lambda_p * j)
            sin_j = np.sin(lambda_p * j)

            # Update the gamma values for the next time step using a recursive process.
            gamma_jtp1[j - 1] = (
                gamma_jt[j - 1] * cos_j
                + gamma_star_jt[j - 1] * sin_j
                + noise_std * np.random.randn()
            )
            gamma_star_jtp1[j - 1] = (
                -gamma_jt[j - 1] * sin_j
                + gamma_star_jt[j - 1] * cos_j
                + noise_std * np.random.randn()
            )

        # Assign the new gamma values to the time series for the current time step.
        series[t] = np.sum(gamma_jtp1)

        # Move to the next time step.
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1

    # After generating the series, discard the burn-in period and return the desired portion of the time series.
    wanted_series = series[-duration:]  # Discard burn-in phase

    return wanted_series


def generate_time_series(
    duration: int = 300,
    periodicities: np.ndarray = np.array([10, 30, 50]),
    num_harmonics: np.ndarray = np.array([3, 2, 2]),
    std: np.ndarray = np.array([2, 3, 5]),
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generates a time series data by combining multiple seasonal components with different periodicities,
    numbers of harmonics, and noise levels. Each component represents a periodic signal, and the final
    series is the sum of these components.

    :param duration: The length of the time series to generate, in terms of number of time steps.
    :param periodicities: An array of the periodicities (number of time steps per cycle) for each seasonal component.
    :param num_harmonics: An array specifying the number of harmonics (sinusoidal components) to use for each seasonal component.
    :param std: An array of standard deviations of the noise for each seasonal component. The noise is added to each harmonic.
    :param seed: A random seed for reproducibility. If None, a random seed is used.
    :return: A numpy array representing the generated time series that combines all the seasonal components.
    """

    # Set the random seed for reproducibility of the results (if provided).
    np.random.seed(seed=seed)

    # List to store the seasonal components that will be combined to form the final series.
    terms = []

    # Loop through each input periodicity and generate a seasonal time series for each.
    for ix, _ in enumerate(periodicities):
        # For each periodicity, generate a seasonal time series using the simulate_seasonal_term function.
        s = simulate_seasonal_term(
            periodicity=periodicities[
                ix
            ],  # The length of the cycle for this component.
            total_cycles=duration
            / periodicities[
                ix
            ],  # Number of cycles to generate based on the total duration.
            harmonics=num_harmonics[ix],  # Number of harmonics for this component.
            noise_std=std[ix],  # Standard deviation of noise for this component.
        )
        # Append the generated seasonal component to the list of terms.
        terms.append(s)

    # Combine all the seasonal components (terms) by summing them together element-wise.
    series = np.sum(terms, axis=0)

    # Return the final time series, which is a combination of the different seasonal components.
    return series
