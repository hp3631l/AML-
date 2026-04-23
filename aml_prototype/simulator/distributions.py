"""
Distribution functions for amount, delay, and frequency sampling.

Used by motif generators to produce realistic transaction parameters.
"""

import numpy as np
from typing import Tuple


def sample_amount(distribution: str, amount_range: Tuple[float, float]) -> float:
    """
    Sample a transaction amount from the specified distribution.

    Args:
        distribution: 'uniform', 'lognormal', 'just_below_threshold'.
        amount_range: (min_amount, max_amount).

    Returns:
        Sampled amount clipped to range.
    """
    lo, hi = amount_range

    if distribution == "lognormal":
        # Log-normal with mean ≈ midpoint of range
        mu = np.log((lo + hi) / 2)
        sigma = 0.5
        amount = np.random.lognormal(mu, sigma)
    elif distribution == "just_below_threshold":
        # Cluster just below $10,000 (CTR threshold)
        threshold = min(hi, 9999.0)
        amount = threshold - abs(np.random.normal(0, 500))
    else:  # uniform
        amount = np.random.uniform(lo, hi)

    return float(np.clip(amount, lo, hi))


def sample_delay_hours(distribution: str, min_hours: float, max_hours: float) -> float:
    """
    Sample inter-transaction delay in hours.

    Args:
        distribution: 'uniform', 'exponential', 'human_mimicking'.
        min_hours: Minimum delay.
        max_hours: Maximum delay.

    Returns:
        Delay in hours.
    """
    if distribution == "exponential":
        # Exponential with scale = midpoint
        scale = (min_hours + max_hours) / 2
        delay = np.random.exponential(scale)
    elif distribution == "human_mimicking":
        # Bimodal: cluster around business hours and late night
        if np.random.random() < 0.7:
            # Business hours gap: 4-12 hours
            delay = np.random.normal((min_hours + max_hours) / 2, max_hours / 4)
        else:
            # Long gap: 24-72 hours
            delay = np.random.uniform(24, max(72, max_hours))
    else:  # uniform
        delay = np.random.uniform(min_hours, max_hours)

    return float(np.clip(delay, min_hours, max_hours))


def sample_retention_rate(low: float = 0.70, high: float = 0.95) -> float:
    """Sample a peel-off retention rate."""
    return float(np.random.uniform(low, high))


def sample_frequency(base_freq: float, noise: float = 0.2) -> float:
    """
    Sample transaction frequency (transactions per day).

    Args:
        base_freq: Base frequency.
        noise: Noise scale as fraction of base.

    Returns:
        Frequency > 0.
    """
    freq = base_freq + np.random.normal(0, base_freq * noise)
    return max(0.01, float(freq))
