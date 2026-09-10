"""
Comparison of all Monte Carlo π estimation methods
"""

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import threading
from multiprocessing import Pool, cpu_count


# ==================== Sequential ====================
def estimate_pi_sequential(n_samples):
    """Sequential: pure Python loop running on a single CPU core (baseline)"""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


# ==================== Vectorized (NumPy) ====================
def estimate_pi_vectorized(n_samples):
    """Vectorized (NumPy): bulk array operations executed in optimized C code on one core"""
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)
    distances_squared = x**2 + y**2
    inside_circle = np.sum(distances_squared <= 1.0)
    return 4.0 * inside_circle / n_samples


# ==================== Numba (JIT) ====================
@jit(nopython=True)
def estimate_pi_numba(n_samples):
    """Numba (JIT): Python loop compiled to native machine code (single-threaded)"""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


# ==================== Threading ====================
def estimate_pi_worker_thread(n_samples, result, index):
    """Worker function for threading"""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    result[index] = inside_circle


def estimate_pi_threading(n_samples, n_threads=4):
    """Threading: multiple Python threads sharing memory, limited by the GIL"""
    samples_per_thread = n_samples // n_threads
    result = [0] * n_threads
    threads = []

    for i in range(n_threads):
        thread = threading.Thread(target=estimate_pi_worker_thread,
                                  args=(samples_per_thread, result, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_inside = sum(result)
    total_samples = samples_per_thread * n_threads
    return 4.0 * total_inside / total_samples


# ==================== Numba Parallel ====================
@jit(nopython=True, parallel=True)
def estimate_pi_numba_parallel(n_samples):
    """Numba Parallel: JIT-compiled loop using prange to distribute work across multiple CPU cores"""
    inside_circle = 0
    for _ in prange(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


# ==================== Multiprocessing ====================
def estimate_pi_worker_process(n_samples):
    """Worker function for multiprocessing"""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return inside_circle


def estimate_pi_multiprocessing(n_samples, n_processes=None):
    """Multiprocessing: multiple independent Python processes with inter-process communication"""
    if n_processes is None:
        n_processes = cpu_count()

    samples_per_process = n_samples // n_processes

    with Pool(processes=n_processes) as pool:
        results = pool.map(estimate_pi_worker_process, [samples_per_process] * n_processes)

    total_inside = sum(results)
    total_samples = samples_per_process * n_processes
    return 4.0 * total_inside / total_samples


# ==================== Benchmark Functions ====================
def warmup_jit():
    """Warmup JIT compiled functions"""
    print("Warming up JIT functions...")
    estimate_pi_numba(1000)
    estimate_pi_numba_parallel(1000)
    print("Warmup complete!\n")


def benchmark_method(name, func, n_samples, *args):
    """Benchmark a single method"""
    start_time = time.time()
    pi_est = func(n_samples, *args)
    elapsed_time = time.time() - start_time
    error = abs(pi_est - np.pi)
    return elapsed_time, error


def run_comparison():
    """Run comparison of all methods"""
    # Warmup JIT functions
    warmup_jit()

    # Sample sizes to test
    sample_sizes = [100_000, 1_000_000, 10_000_000]

    # Methods to compare
    methods = [
        ("Sequential", estimate_pi_sequential, []),
        ("Vectorized (NumPy)", estimate_pi_vectorized, []),
        ("Numba JIT", estimate_pi_numba, []),
        ("Threading (4 threads)", estimate_pi_threading, [4]),
        ("Numba Parallel", estimate_pi_numba_parallel, []),
        ("Multiprocessing", estimate_pi_multiprocessing, [cpu_count()]),
    ]

    # Store results
    results = {method[0]: {"times": [], "errors": []} for method in methods}

    # Run benchmarks
    print("Running benchmarks...")
    print("=" * 80)

    for n_samples in sample_sizes:
        print(f"\nSample size: {n_samples:,}")
        print("-" * 80)

        for method_name, func, args in methods:
            try:
                elapsed_time, error = benchmark_method(method_name, func, n_samples, *args)
                results[method_name]["times"].append(elapsed_time)
                results[method_name]["errors"].append(error)
                print(f"{method_name:<30} Time: {elapsed_time:>8.4f}s  Error: {error:.10f}")
            except Exception as e:
                print(f"{method_name:<30} Error: {str(e)}")
                results[method_name]["times"].append(None)
                results[method_name]["errors"].append(None)

    print("\n" + "=" * 80)

    return results, sample_sizes, methods


def plot_results(results, sample_sizes, methods):
    """Create visualization of results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Execution Time Comparison
    x = np.arange(len(sample_sizes))
    width = 0.13

    for i, (method_name, _, _) in enumerate(methods):
        times = results[method_name]["times"]
        if None not in times:
            ax1.bar(x + i * width, times, width, label=method_name)

    ax1.set_xlabel('Sample Size', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels([f'{n:,}' for n in sample_sizes])
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Speedup vs Sequential (for largest sample size)
    baseline_time = results["Sequential"]["times"][-1]
    speedups = []
    method_names = []

    for method_name, _, _ in methods:
        times = results[method_name]["times"]
        if None not in times and times[-1] > 0:
            speedup = baseline_time / times[-1]
            speedups.append(speedup)
            method_names.append(method_name)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax2.barh(method_names, speedups, color=colors[:len(method_names)])

    # Add speedup values on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        ax2.text(speedup + 0.5, i, f'{speedup:.2f}x',
                va='center', fontweight='bold', fontsize=10)

    ax2.set_xlabel('Speedup vs Sequential', fontsize=12)
    ax2.set_title(f'Speedup Comparison ({sample_sizes[-1]:,} samples)',
                 fontsize=14, fontweight='bold')
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend(fontsize=10)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = r'c:\Users\xl\OneDrive - Universitat Ramón Llull\git\practice\monte_carlo_pi_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{save_path}'")
    plt.close()


def print_summary_table(results, sample_sizes, methods):
    """Print a summary table"""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (10,000,000 samples)")
    print("=" * 100)
    print(f"{'Method':<30} {'Time (s)':<12} {'Speedup':<12} {'Error':<15}")
    print("-" * 100)

    baseline_time = results["Sequential"]["times"][-1]

    for method_name, _, _ in methods:
        time_val = results[method_name]["times"][-1]
        error_val = results[method_name]["errors"][-1]

        if time_val is not None:
            speedup = baseline_time / time_val
            print(f"{method_name:<30} {time_val:<12.4f} {speedup:<12.2f}x {error_val:<15.10f}")

    print("=" * 100)


def main():
    print("Monte Carlo π Estimation - Method Comparison")
    print("=" * 80)
    print(f"CPU cores available: {cpu_count()}")
    print()

    # Run comparison
    results, sample_sizes, methods = run_comparison()

    # Print summary
    print_summary_table(results, sample_sizes, methods)

    # Create visualization
    plot_results(results, sample_sizes, methods)


if __name__ == "__main__":
    main()
