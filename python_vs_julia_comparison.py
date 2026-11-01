"""
Python vs Julia Monte Carlo π Estimation Comparison
Run both implementations and compare results
"""

import subprocess
import time
import random
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt


# ==================== Python Implementations ====================

def estimate_pi_python_sequential(n_samples):
    """Python Sequential"""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


def estimate_pi_python_numpy(n_samples):
    """Python NumPy Vectorized"""
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)
    inside_circle = np.sum((x**2 + y**2) <= 1.0)
    return 4.0 * inside_circle / n_samples


@jit(nopython=True)
def estimate_pi_python_numba(n_samples):
    """Python Numba JIT"""
    inside_circle = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


@jit(nopython=True, parallel=True)
def estimate_pi_python_numba_parallel(n_samples):
    """Python Numba Parallel"""
    inside_circle = 0
    for _ in prange(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside_circle += 1
    return 4.0 * inside_circle / n_samples


def check_julia_installed():
    """Check if Julia is installed"""
    try:
        result = subprocess.run(['julia', '--version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        return result.returncode == 0
    except:
        return False


def run_python_benchmarks(sample_size):
    """Run Python benchmarks"""
    print(f"\nPython Benchmarks ({sample_size:,} samples):")
    print("-" * 70)

    results = {}

    # Warmup Numba
    estimate_pi_python_numba(1000)
    estimate_pi_python_numba_parallel(1000)

    methods = [
        ("Python Sequential", estimate_pi_python_sequential),
        ("Python NumPy", estimate_pi_python_numpy),
        ("Python Numba JIT", estimate_pi_python_numba),
        ("Python Numba Parallel", estimate_pi_python_numba_parallel),
    ]

    for name, func in methods:
        start = time.time()
        pi_est = func(sample_size)
        elapsed = time.time() - start
        error = abs(pi_est - np.pi)

        results[name] = {"time": elapsed, "error": error, "pi": pi_est}
        print(f"{name:<25} Time: {elapsed:>8.4f}s  Error: {error:.10f}")

    return results


def run_julia_benchmarks(sample_size):
    """Run Julia benchmarks by calling the Julia script"""
    if not check_julia_installed():
        print("\nJulia is not installed or not in PATH. Skipping Julia benchmarks.")
        print("To install Julia: https://julialang.org/downloads/")
        return None

    print(f"\nJulia Benchmarks ({sample_size:,} samples):")
    print("-" * 70)

    # Create a temporary Julia script for this specific sample size
    julia_script = f'''
    using Printf

    function estimate_pi_sequential(n)
        inside = 0
        for _ in 1:n
            x, y = rand(), rand()
            if x^2 + y^2 <= 1.0
                inside += 1
            end
        end
        return 4.0 * inside / n
    end

    function estimate_pi_vectorized(n)
        x = rand(n)
        y = rand(n)
        return 4.0 * sum((x.^2 .+ y.^2) .<= 1.0) / n
    end

    function estimate_pi_threads(n)
        inside = Threads.Atomic{{Int}}(0)
        Threads.@threads for _ in 1:n
            x, y = rand(), rand()
            if x^2 + y^2 <= 1.0
                Threads.atomic_add!(inside, 1)
            end
        end
        return 4.0 * inside[] / n
    end

    n = {sample_size}

    # Warmup
    estimate_pi_sequential(1000)
    estimate_pi_vectorized(1000)
    estimate_pi_threads(1000)

    # Benchmark
    methods = [
        ("Julia Sequential", estimate_pi_sequential),
        ("Julia Vectorized", estimate_pi_vectorized),
        ("Julia Multi-threading", estimate_pi_threads),
    ]

    for (name, func) in methods
        t = @elapsed pi_est = func(n)
        err = abs(pi_est - π)
        @printf("%-25s Time: %8.4fs  Error: %.10f\\n", name, t, err)
    end
    '''

    try:
        # Write temp script
        with open('temp_julia_bench.jl', 'w') as f:
            f.write(julia_script)

        # Run Julia script
        result = subprocess.run(['julia', '--threads=auto', 'temp_julia_bench.jl'],
                              capture_output=True,
                              text=True,
                              timeout=120)

        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Julia error: {result.stderr}")

        # Clean up
        import os
        if os.path.exists('temp_julia_bench.jl'):
            os.remove('temp_julia_bench.jl')

        return True

    except Exception as e:
        print(f"Error running Julia: {e}")
        return None


def main():
    print("="*70)
    print("Python vs Julia Monte Carlo π Estimation Comparison")
    print("="*70)

    sample_sizes = [1_000_000, 10_000_000]

    for sample_size in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Sample Size: {sample_size:,}")
        print(f"{'='*70}")

        # Run Python benchmarks
        python_results = run_python_benchmarks(sample_size)

        # Run Julia benchmarks
        run_julia_benchmarks(sample_size)

        print()

    print("="*70)
    print(f"Actual π = {np.pi:.11f}")
    print("="*70)

    print("\nKey Insights:")
    print("- Julia Sequential is typically faster than Python Sequential (no GIL)")
    print("- Python Numba Parallel is often competitive with Julia Multi-threading")
    print("- Julia's native performance is excellent without JIT compilation")
    print("- Both languages excel at parallel Monte Carlo simulations")


if __name__ == "__main__":
    main()
