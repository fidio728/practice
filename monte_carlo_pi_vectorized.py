"""
Vectorized Monte Carlo estimation of π using NumPy
Bulk array operations executed in optimized C code on one core
"""

import numpy as np
import time


def estimate_pi_vectorized(n_samples):
    """
    Estimate π using Monte Carlo method with vectorized NumPy operations.

    Args:
        n_samples: Number of random points to generate

    Returns:
        Estimated value of π
    """
    # Generate random points in unit square [0, 1] × [0, 1]
    # All at once using NumPy's vectorized operations
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)

    # Check if points are inside quarter circle (x² + y² ≤ 1)
    # Vectorized operation - no Python loop!
    distances_squared = x**2 + y**2
    inside_circle = np.sum(distances_squared <= 1.0)

    # π ≈ 4 × (points inside circle / total points)
    pi_estimate = 4.0 * inside_circle / n_samples

    return pi_estimate


def main():
    # Test with different sample sizes
    sample_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    print("Vectorized (NumPy) Monte Carlo π Estimation")
    print("=" * 60)
    print(f"{'Samples':<15} {'Estimated π':<15} {'Error':<12} {'Time (s)':<10}")
    print("-" * 60)

    for n_samples in sample_sizes:
        start_time = time.time()
        pi_est = estimate_pi_vectorized(n_samples)
        elapsed_time = time.time() - start_time

        error = abs(pi_est - np.pi)

        print(f"{n_samples:<15,} {pi_est:<15.10f} {error:<12.10f} {elapsed_time:<10.4f}")

    print("=" * 60)
    print(f"Actual π = {np.pi:.11f}")


if __name__ == "__main__":
    main()
