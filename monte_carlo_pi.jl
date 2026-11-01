# Monte Carlo π Estimation in Julia
# Comparing different parallelization strategies

using Statistics
using Printf


# ==================== Sequential ====================
function estimate_pi_sequential(n_samples::Int)
    inside_circle = 0

    for _ in 1:n_samples
        x = rand()
        y = rand()
        if x^2 + y^2 <= 1.0
            inside_circle += 1
        end
    end

    return 4.0 * inside_circle / n_samples
end


# ==================== Multi-threading ====================
function estimate_pi_threads(n_samples::Int)
    inside_circle = Threads.Atomic{Int}(0)

    Threads.@threads for _ in 1:n_samples
        x = rand()
        y = rand()
        if x^2 + y^2 <= 1.0
            Threads.atomic_add!(inside_circle, 1)
        end
    end

    return 4.0 * inside_circle[] / n_samples
end


# ==================== Vectorized ====================
function estimate_pi_vectorized(n_samples::Int)
    x = rand(n_samples)
    y = rand(n_samples)
    inside_circle = sum((x.^2 .+ y.^2) .<= 1.0)
    return 4.0 * inside_circle / n_samples
end


# ==================== Benchmark Function ====================
function benchmark_method(func::Function, n_samples::Int)
    # Warmup
    func(1000)

    # Actual benchmark
    elapsed_time = @elapsed pi_est = func(n_samples)
    error = abs(pi_est - π)

    return elapsed_time, error, pi_est
end


# ==================== Main ====================
function main()
    println("Julia Monte Carlo π Estimation")
    println("="^60)
    println("Julia threads available: ", Threads.nthreads())
    println()

    sample_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    methods = [
        ("Sequential", estimate_pi_sequential),
        ("Vectorized", estimate_pi_vectorized),
        ("Multi-threading", estimate_pi_threads),
    ]

    println(@sprintf("%-15s %-15s %-12s %-10s", "Samples", "Estimated π", "Error", "Time (s)"))
    println("-"^60)

    results = Dict()

    for (name, func) in methods
        println("\n$name:")
        println("="^60)
        println(@sprintf("%-15s %-15s %-12s %-10s", "Samples", "Estimated π", "Error", "Time (s)"))
        println("-"^60)

        results[name] = []

        for n_samples in sample_sizes
            try
                elapsed_time, error, pi_est = benchmark_method(func, n_samples)
                push!(results[name], (n_samples, elapsed_time, error, pi_est))

                samples_str = replace(string(n_samples), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
                @printf("%-15s %-15.10f %-12.10f %-10.4f\n", samples_str, pi_est, error, elapsed_time)
            catch e
                println("Error at $n_samples samples: $e")
            end
        end
    end

    println("\n" * "="^60)
    @printf("Actual π = %.11f\n", π)

    # Print summary comparison
    println("\n" * "="^80)
    println("SUMMARY COMPARISON (10,000,000 samples)")
    println("="^80)
    @printf("%-25s %-12s %-12s %-15s\n", "Method", "Time (s)", "Speedup", "Error")
    println("-"^80)

    # Get baseline time (sequential)
    baseline_time = 0.0
    for (name, _) in methods
        if name == "Sequential"
            for (n, t, _, _) in results[name]
                if n == 10_000_000
                    baseline_time = t
                    break
                end
            end
            break
        end
    end

    for (name, _) in methods
        for (n, t, e, _) in results[name]
            if n == 10_000_000
                speedup = baseline_time / t
                @printf("%-25s %-12.4f %-12.2fx %-15.10f\n", name, t, speedup, e)
                break
            end
        end
    end

    println("="^80)
end

# Run the main function
main()