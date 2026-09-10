"""
Numba JIT Monte Carlo estimation of π
Python loop compiled to native machine code (single-threaded)
"""

import random
import time
from numba import jit


@jit(nopython=True)
def estimate_pi_numba(n_samples):
    """
    Estimate π using Monte Carlo method with Numba JIT compilation.
    This Python loop will be compiled to native machine code.

    Args:
        n_samples: Number of random points to generate

    Returns:
        Estimated value of π
    """
    inside_circle = 0

    for _ in range(n_samples):
        # Generate random point in unit square [0, 1] × [0, 1]
        x = random.random()
        y = random.random()

        # Check if point is inside quarter circle (x² + y² ≤ 1)
        if x*x + y*y <= 1.0:
            inside_circle += 1

    # π ≈ 4 × (points inside circle / total points)
    pi_estimate = 4.0 * inside_circle / n_samples

    return pi_estimate


def main():
    # Warmup to compile the JIT function
    print("Warming up JIT compilation...")
    _ = estimate_pi_numba(1000)
    print("JIT compilation complete!\n")

    # Test with different sample sizes
    sample_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    print("Numba JIT Monte Carlo π Estimation")
    print("=" * 60)
    print(f"{'Samples':<15} {'Estimated π':<15} {'Error':<12} {'Time (s)':<10}")
    print("-" * 60)

    for n_samples in sample_sizes:
        start_time = time.time()
        pi_est = estimate_pi_numba(n_samples)
        elapsed_time = time.time() - start_time

        error = abs(pi_est - 3.14159265359)

        print(f"{n_samples:<15,} {pi_est:<15.10f} {error:<12.10f} {elapsed_time:<10.4f}")

    print("=" * 60)
    print(f"Actual π = 3.14159265359")


if __name__ == "__main__":
    main()
