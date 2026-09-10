"""
Multiprocessing Monte Carlo estimation of π
Multiple independent Python processes running in parallel with inter-process communication
"""

import random
import time
from multiprocessing import Pool, cpu_count


def estimate_pi_worker(n_samples):
    """
    Worker function for a single process to estimate π.

    Args:
        n_samples: Number of random points to generate

    Returns:
        Count of points inside the quarter circle
    """
    inside_circle = 0

    for _ in range(n_samples):
        # Generate random point in unit square [0, 1] × [0, 1]
        x = random.random()
        y = random.random()

        # Check if point is inside quarter circle (x² + y² ≤ 1)
        if x*x + y*y <= 1.0:
            inside_circle += 1

    return inside_circle


def estimate_pi_multiprocessing(n_samples, n_processes=None):
    """
    Estimate π using Monte Carlo method with multiple processes.

    Args:
        n_samples: Total number of random points to generate
        n_processes: Number of processes to use (default: CPU count)

    Returns:
        Estimated value of π
    """
    if n_processes is None:
        n_processes = cpu_count()

    # Divide work among processes
    samples_per_process = n_samples // n_processes

    # Create a pool of worker processes
    with Pool(processes=n_processes) as pool:
        # Distribute work to processes
        results = pool.map(estimate_pi_worker, [samples_per_process] * n_processes)

    # Combine results from all processes
    total_inside = sum(results)
    total_samples = samples_per_process * n_processes

    # π ≈ 4 × (points inside circle / total points)
    pi_estimate = 4.0 * total_inside / total_samples

    return pi_estimate


def main():
    # Test with different sample sizes
    sample_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    n_processes = cpu_count()

    print(f"Multiprocessing Monte Carlo π Estimation (using {n_processes} processes)")
    print("=" * 60)
    print(f"{'Samples':<15} {'Estimated π':<15} {'Error':<12} {'Time (s)':<10}")
    print("-" * 60)

    for n_samples in sample_sizes:
        start_time = time.time()
        pi_est = estimate_pi_multiprocessing(n_samples, n_processes)
        elapsed_time = time.time() - start_time

        error = abs(pi_est - 3.14159265359)

        print(f"{n_samples:<15,} {pi_est:<15.10f} {error:<12.10f} {elapsed_time:<10.4f}")

    print("=" * 60)
    print(f"Actual π = 3.14159265359")
    print(f"\nNote: Each process runs independently with its own Python interpreter,")
    print(f"avoiding the GIL and enabling true parallel execution.")


if __name__ == "__main__":
    main()
