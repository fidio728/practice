"""
Multiprocessing Monte Carlo estimation of π with CPU monitoring
Multiple independent Python processes running in parallel with inter-process communication
"""

import random
import time
import psutil
import threading
from multiprocessing import Pool, cpu_count


# Global variables for CPU monitoring
cpu_percentages = []
monitoring = False


def monitor_cpu():
    """Monitor CPU usage in a separate thread"""
    global cpu_percentages, monitoring
    while monitoring:
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
        cpu_percentages.append(cpu_percent)


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
        Estimated value of π, average CPU usage, peak CPU usage
    """
    global cpu_percentages, monitoring

    if n_processes is None:
        n_processes = cpu_count()

    # Divide work among processes
    samples_per_process = n_samples // n_processes

    # Start CPU monitoring
    cpu_percentages = []
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_cpu)
    monitor_thread.start()

    # Create a pool of worker processes
    with Pool(processes=n_processes) as pool:
        # Distribute work to processes
        results = pool.map(estimate_pi_worker, [samples_per_process] * n_processes)

    # Stop CPU monitoring
    monitoring = False
    monitor_thread.join()

    # Combine results from all processes
    total_inside = sum(results)
    total_samples = samples_per_process * n_processes

    # π ≈ 4 × (points inside circle / total points)
    pi_estimate = 4.0 * total_inside / total_samples

    # Calculate CPU statistics
    if cpu_percentages:
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
        peak_cpu = max(cpu_percentages)
    else:
        avg_cpu = 0
        peak_cpu = 0

    return pi_estimate, avg_cpu, peak_cpu


def main():
    # Test with different sample sizes
    sample_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    n_processes = cpu_count()

    print(f"Multiprocessing Monte Carlo π Estimation (using {n_processes} processes)")
    print(f"System has {psutil.cpu_count(logical=True)} logical CPUs, {psutil.cpu_count(logical=False)} physical cores")
    print("=" * 90)
    print(f"{'Samples':<15} {'Estimated π':<15} {'Error':<12} {'Time (s)':<10} {'Avg CPU%':<10} {'Peak CPU%':<10}")
    print("-" * 90)

    for n_samples in sample_sizes:
        start_time = time.time()
        pi_est, avg_cpu, peak_cpu = estimate_pi_multiprocessing(n_samples, n_processes)
        elapsed_time = time.time() - start_time

        error = abs(pi_est - 3.14159265359)

        print(f"{n_samples:<15,} {pi_est:<15.10f} {error:<12.10f} {elapsed_time:<10.4f} {avg_cpu:<10.1f} {peak_cpu:<10.1f}")

    print("=" * 90)
    print(f"Actual π = 3.14159265359")
    print(f"\nNote: Each process runs independently with its own Python interpreter,")
    print(f"avoiding the GIL and enabling true parallel execution.")
    print(f"CPU usage shows utilization across all {n_processes} cores.")


if __name__ == "__main__":
    main()