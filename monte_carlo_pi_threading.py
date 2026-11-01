"""
Threading Monte Carlo estimation of π
Multiple Python threads sharing memory, limited by the Global Interpreter Lock (GIL)
"""

import random
import time
import threading


def estimate_pi_worker(n_samples, result, index):
    """
    Worker function for a single thread to estimate π.

    Args:
        n_samples: Number of random points to generate
        result: Shared list to store the count of points inside circle
        index: Index in the result list for this thread
    """
    inside_circle = 0

    for _ in range(n_samples):
        # Generate random point in unit square [0, 1] × [0, 1]
        x = random.random()
        y = random.random()

        # Check if point is inside quarter circle (x² + y² ≤ 1)
        if x*x + y*y <= 1.0:
            inside_circle += 1

    result[index] = inside_circle


def estimate_pi_threading(n_samples, n_threads=4):
    """
    Estimate π using Monte Carlo method with multiple threads.

    Args:
        n_samples: Total number of random points to generate
        n_threads: Number of threads to use

    Returns:
        Estimated value of π
    """
    # Divide work among threads
    samples_per_thread = n_samples // n_threads
    result = [0] * n_threads
    threads = []

    # Create and start threads
    for i in range(n_threads):
        thread = threading.Thread(
            target=estimate_pi_worker,
            args=(samples_per_thread, result, i)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Combine results from all threads
    total_inside = sum(result)
    total_samples = samples_per_thread * n_threads

    # π ≈ 4 × (points inside circle / total points)
    pi_estimate = 4.0 * total_inside / total_samples

    return pi_estimate


def main():
    # Test with different sample sizes
    sample_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    n_threads = 4

    print(f"Threading Monte Carlo π Estimation (using {n_threads} threads)")
    print("=" * 60)
    print(f"{'Samples':<15} {'Estimated π':<15} {'Error':<12} {'Time (s)':<10}")
    print("-" * 60)

    for n_samples in sample_sizes:
        start_time = time.time()
        pi_est = estimate_pi_threading(n_samples, n_threads)
        elapsed_time = time.time() - start_time

        error = abs(pi_est - 3.14159265359)

        print(f"{n_samples:<15,} {pi_est:<15.10f} {error:<12.10f} {elapsed_time:<10.4f}")

    print("=" * 60)
    print(f"Actual π = 3.14159265359")
    print(f"\nNote: Due to Python's GIL, threading provides limited speedup")
    print(f"for CPU-bound tasks like this Monte Carlo simulation.")


if __name__ == "__main__":
    main()
