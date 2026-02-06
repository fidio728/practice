# ============================================================================
# Dynamic Inventory Management - MDP Solution (DISCRETE VERSION)
# Value Iteration, Policy Iteration, and Monte Carlo Methods
#
# Authors: Saba & Xueyang
# Course: SOCM Assignment 3
# VERSION: DISCRETE - Integer states and actions, discrete demand
# ============================================================================

# ============================================================================
# 第一部分: 导入包
# ============================================================================

using Plots
using Printf
using Statistics
using Random
g   
# 设置随机种子（可选，用于结果可复现）
Random.seed!(42)

# ============================================================================
# 第二部分: 数据结构定义 (DISCRETE)
# ============================================================================

struct InventoryMDP
    h::Float64              # holding cost
    b::Float64              # penalty cost
    c::Float64              # ordering cost
    γ::Float64              # discount factor
    θ::Float64              # convergence threshold
    demand_values::Vector{Int}  # discrete demand values
    demand_prob::Float64    # probability for each demand (uniform)
    states::Vector{Int}     # INTEGER states
    n_states::Int
end

function InventoryMDP(;
    demand_values = [0, 1, 2, 3, 4],  # DISCRETE demand
    h = 1.0,
    b = 9.0,
    c = 0.0,
    γ = 0.95,
    θ = 1e-4,
    x_min = -2,  # INTEGER
    x_max = 8    # INTEGER
)
    states = collect(x_min:x_max)  # Integer range
    n_states = length(states)
    demand_prob = 1.0 / length(demand_values)  # uniform probability

    println("=" ^ 70)
    println("DISCRETE VERSION - Integer states and actions")
    println("State space: [$x_min, $x_max] with $n_states integer states")
    println("Demand distribution: Discrete Uniform on $demand_values")
    println("Demand probability: $demand_prob each")
    println("Parameters: h=$h, b=$b, c=$c, γ=$γ")
    println("=" ^ 70)

    return InventoryMDP(h, b, c, γ, θ, demand_values, demand_prob, states, n_states)
end

# ============================================================================
# 第三部分: 核心计算函数 (DISCRETE)
# ============================================================================

function immediate_cost(mdp::InventoryMDP, y::Int)
    (; h, b, demand_values, demand_prob) = mdp

    expected_holding = 0.0
    expected_stockout = 0.0

    # Discrete summation over all demand values
    for d in demand_values
        # Holding cost: positive inventory after demand
        holding = max(0, y - d)
        expected_holding += holding * demand_prob

        # Stockout cost: unmet demand
        stockout = max(0, d - y)
        expected_stockout += stockout * demand_prob
    end

    return h * expected_holding + b * expected_stockout
end

function get_value_at_state(mdp::InventoryMDP, x::Int, V::Vector{Float64})
    (; states) = mdp

    # Boundary handling
    if x < states[1]
        return V[1]
    end
    if x > states[end]
        return V[end]
    end

    # Direct lookup for integer states
    idx = x - states[1] + 1
    return V[idx]
end

function expected_next_value(mdp::InventoryMDP, y::Int, V::Vector{Float64})
    (; demand_values, demand_prob) = mdp

    expected_value = 0.0

    # Discrete summation: E[V(y-D)] = sum over all d of V(y-d) * P(D=d)
    for d in demand_values
        next_state = y - d
        next_value = get_value_at_state(mdp, next_state, V)
        expected_value += next_value * demand_prob
    end

    return expected_value
end

function q_function(mdp::InventoryMDP, x::Int, y::Int, V::Vector{Float64})
    ordering_cost = mdp.c * (y - x)
    holding_penalty = immediate_cost(mdp, y)
    future_value = mdp.γ * expected_next_value(mdp, y, V)

    return -ordering_cost - holding_penalty + future_value
end

# ============================================================================
# 第四部分: Value Iteration (DISCRETE)
# ============================================================================

function value_iteration(mdp::InventoryMDP; max_iterations=1000, verbose=true)
    (; states, n_states, θ) = mdp

    V = zeros(n_states)
    policy = zeros(Int, n_states)  # INTEGER policy
    convergence_history = Float64[]

    if verbose
        println("\n" * "=" ^ 70)
        println("VALUE ITERATION (DISCRETE)")
        println("=" ^ 70)
    end

    iteration = 0
    for iter in 1:max_iterations
        iteration = iter
        V_old = copy(V)

        for i in 1:n_states
            x = states[i]

            # Action space: integers y >= x
            y_max = x + 10
            y_values = collect(x:y_max)  # INTEGER range

            q_values = [q_function(mdp, x, y, V_old) for y in y_values]

            best_idx = argmax(q_values)
            V[i] = q_values[best_idx]
            policy[i] = y_values[best_idx]
        end

        δ = maximum(abs.(V - V_old))
        push!(convergence_history, δ)

        if verbose && iter % 10 == 0
            @printf("  Iteration %3d: Δ = %.8f\n", iter, δ)
        end

        if δ < θ
            if verbose
                @printf("  Converged after %d iterations (Δ = %.8f)\n", iter, δ)
            end
            break
        end
    end

    return V, policy, iteration, convergence_history
end

# ============================================================================
# 第五部分: Policy Iteration (Iterative Evaluation)
# ============================================================================

function policy_evaluation(mdp::InventoryMDP, policy::Vector{Int}, V::Vector{Float64};
                           max_iterations=100, verbose=false)
    (; states, n_states, c, γ, θ) = mdp

    for iter in 1:max_iterations
        V_old = copy(V)

        for i in 1:n_states
            x = states[i]
            y = policy[i]

            ordering_cost = c * (y - x)
            holding_penalty = immediate_cost(mdp, y)
            future_value = γ * expected_next_value(mdp, y, V_old)

            V[i] = -ordering_cost - holding_penalty + future_value
        end

        δ = maximum(abs.(V - V_old))

        if verbose && iter % 20 == 0
            @printf("    Policy Eval Iteration %3d: Δ = %.8f\n", iter, δ)
        end

        if δ < θ
            if verbose
                @printf("    Policy Eval converged after %d iterations\n", iter)
            end
            return iter
        end
    end

    return max_iterations
end

function policy_improvement(mdp::InventoryMDP, V::Vector{Float64}, policy::Vector{Int};
                            verbose=false)
    (; states, n_states) = mdp

    policy_old = copy(policy)

    for i in 1:n_states
        x = states[i]

        y_max = x + 10
        y_values = collect(x:y_max)  # INTEGER range

        q_values = [q_function(mdp, x, y, V) for y in y_values]
        policy[i] = y_values[argmax(q_values)]
    end

    n_changed = sum(policy .!= policy_old)
    policy_stable = (n_changed == 0)

    if verbose
        println("  Policy Improvement: $n_changed states changed")
    end

    return policy_stable
end

function policy_iteration(mdp::InventoryMDP; max_iterations=100, verbose=true)
    (; states, n_states, demand_values) = mdp

    V = zeros(n_states)
    # Initialize with a simple policy: order up to max demand
    policy = [max(x, maximum(demand_values)) for x in states]

    if verbose
        println("\n" * "=" ^ 70)
        println("POLICY ITERATION (ITERATIVE - DISCRETE)")
        println("=" ^ 70)
    end

    for iter in 1:max_iterations
        if verbose
            println("\n--- Iteration $iter ---")
            println("  Step 1: Policy Evaluation")
        end

        eval_iters = policy_evaluation(mdp, policy, V, max_iterations=100, verbose=verbose)

        if verbose
            println("  Step 2: Policy Improvement")
        end
        policy_stable = policy_improvement(mdp, V, policy, verbose=verbose)

        if policy_stable
            if verbose
                println("\nPolicy converged after $iter iterations")
            end
            return V, policy, iter
        end
    end

    if verbose
        println("\nReached maximum iterations ($max_iterations)")
    end
    return V, policy, max_iterations
end

# ============================================================================
# 第六部分: Monte Carlo Policy Evaluation
# ============================================================================

function generate_episode(mdp::InventoryMDP, policy::Vector{Int},
                          x0::Int, N::Int, v_terminal::Function)
    (; c, h, b, demand_values, states) = mdp

    state_list = Int[]
    action_list = Int[]
    reward_list = Float64[]

    x = x0

    for t in 1:N
        push!(state_list, x)

        idx = findfirst(==(x), states)
        if idx === nothing
            idx = argmin(abs.(states .- x))
        end
        y = policy[idx]

        push!(action_list, y)

        # Sample discrete demand
        D = rand(demand_values)

        ordering_cost = c * (y - x)

        inventory_after_demand = y - D
        if inventory_after_demand >= 0
            holding_cost = h * inventory_after_demand
            shortage_cost = 0.0
        else
            holding_cost = 0.0
            shortage_cost = b * (-inventory_after_demand)
        end

        reward = -(ordering_cost + holding_cost + shortage_cost)
        push!(reward_list, reward)

        x = y - D
        x = clamp(x, states[1], states[end])
    end

    terminal_value = v_terminal(x)
    push!(reward_list, terminal_value)

    return state_list, action_list, reward_list
end

function first_visit_mc_evaluation(mdp::InventoryMDP, policy::Vector{Int},
                                    N::Int, v_terminal::Function;
                                    n_episodes::Int=1000, verbose::Bool=true)
    (; states, n_states, γ) = mdp

    Returns = [Float64[] for _ in 1:n_states]

    if verbose
        println("    Running $n_episodes episodes...")
    end

    for episode in 1:n_episodes
        x0 = states[rand(1:n_states)]

        state_list, action_list, reward_list = generate_episode(
            mdp, policy, x0, N, v_terminal
        )

        T = length(state_list)
        G = 0.0

        visited_states = Set{Int}()

        for t in T:-1:1
            if t == T
                G = reward_list[t] + γ * reward_list[t + 1]
            else
                G = reward_list[t + 1] + γ * G
            end

            s = state_list[t]
            state_idx = findfirst(==(s), states)
            if state_idx === nothing
                state_idx = argmin(abs.(states .- s))
            end

            if !(state_idx in visited_states)
                push!(visited_states, state_idx)
                push!(Returns[state_idx], G)
            end
        end

        if verbose && episode % (n_episodes ÷ 5) == 0
            @printf("      Episode %d / %d\n", episode, n_episodes)
        end
    end

    V = zeros(n_states)
    for i in 1:n_states
        if !isempty(Returns[i])
            V[i] = mean(Returns[i])
        end
    end

    return V
end

function policy_iteration_mc(mdp::InventoryMDP;
                              N::Int=20,
                              n_episodes::Int=2000,
                              max_iterations::Int=20,
                              verbose::Bool=true)
    (; states, n_states, demand_values, c) = mdp

    v_terminal(x) = -c * max(x, 0)

    policy = [max(x, maximum(demand_values)) for x in states]
    V = zeros(n_states)

    if verbose
        println("\n" * "=" ^ 70)
        println("POLICY ITERATION (MONTE CARLO - DISCRETE)")
        println("Horizon N = $N, Episodes = $n_episodes")
        println("=" ^ 70)
    end

    for iter in 1:max_iterations
        if verbose
            println("\n--- Iteration $iter ---")
            println("  Step 1: Monte Carlo Policy Evaluation")
        end

        V = first_visit_mc_evaluation(mdp, policy, N, v_terminal,
                                       n_episodes=n_episodes, verbose=verbose)

        if verbose
            @printf("    V range: [%.4f, %.4f]\n", minimum(V), maximum(V))
            println("  Step 2: Policy Improvement")
        end

        policy_old = copy(policy)

        for i in 1:n_states
            x = states[i]

            y_max = x + 10
            y_values = collect(x:y_max)  # INTEGER range

            q_values = [q_function(mdp, x, y, V) for y in y_values]

            policy[i] = y_values[argmax(q_values)]
        end

        n_changed = sum(policy .!= policy_old)
        policy_stable = (n_changed == 0)

        if verbose
            println("    $n_changed states changed policy")
        end

        if policy_stable
            if verbose
                println("\nPolicy converged after $iter iterations")
            end
            return V, policy, iter
        end
    end

    if verbose
        println("\nReached maximum iterations ($max_iterations)")
    end

    return V, policy, max_iterations
end

# ============================================================================
# 第七部分: 解析解 (DISCRETE - Approximate)
# ============================================================================

function analytical_base_stock(mdp::InventoryMDP)
    (; h, b, c, γ, demand_values) = mdp
    # For discrete version, use approximation based on continuous formula
    d_min = minimum(demand_values)
    d_max = maximum(demand_values)
    critical_ratio = (b - (1 - γ) * c) / (b + h)
    S_star = d_min + (d_max - d_min) * critical_ratio
    # Round to nearest integer for discrete case
    return round(Int, S_star)
end

# ============================================================================
# 第八部分: 可视化
# ============================================================================

function plot_results(mdp::InventoryMDP, V::Vector{Float64}, policy::Vector{Int},
                      title_prefix::String="")
    S_analytical = analytical_base_stock(mdp)

    # Add markers for discrete points
    p1 = plot(mdp.states, V,
        xlabel = "Inventory Level (x)",
        ylabel = "V*(x)",
        title = "$(title_prefix)Value Function (DISCRETE)",
        linewidth = 2,
        legend = false,
        color = :blue,
        marker = :circle,
        markersize = 3
    )

    p2 = plot(mdp.states, policy,
        xlabel = "Inventory Level (x)",
        ylabel = "Order-Up-To Level (y)",
        title = "$(title_prefix)Optimal Policy (DISCRETE)",
        linewidth = 2,
        label = "π*(x)",
        color = :red,
        marker = :square,
        markersize = 3
    )
    plot!(mdp.states, mdp.states,
        linestyle = :dash,
        color = :black,
        alpha = 0.5,
        label = "y = x (no order)"
    )
    hline!([S_analytical],
        linestyle = :dot,
        color = :green,
        linewidth = 2,
        label = @sprintf("S* ≈ %d", S_analytical)
    )

    p = plot(p1, p2, layout=(1, 2), size=(1000, 400))
    return p
end

function print_results(mdp::InventoryMDP, V::Vector{Float64}, policy::Vector{Int},
                       method_name::String)
    S_analytical = analytical_base_stock(mdp)

    println("\n" * "-" ^ 50)
    println("$method_name Results (DISCRETE - Integer Actions):")
    println("-" ^ 50)
    @printf("%-15s %-20s %-15s\n", "State (x)", "Optimal Action (y)", "Value V*(x)")
    println("-" ^ 50)

    step = max(1, mdp.n_states ÷ 10)
    for i in 1:step:mdp.n_states
        @printf("%-15d %-20d %-15.4f\n", mdp.states[i], policy[i], V[i])
    end
    println("-" ^ 50)
    @printf("Numerical S*: %d, Analytical S*: %d, Error: %d\n",
            policy[1], S_analytical, abs(policy[1] - S_analytical))
end

# ============================================================================
# 第九部分: Main 函数
# ============================================================================

function main()
    # ===== 创建 DISCRETE MDP =====
    mdp = InventoryMDP(
        demand_values = [0, 1, 2, 3, 4],  # DISCRETE demand
        h = 1.0,
        b = 9.0,
        c = 0.0,
        γ = 0.95,
        θ = 1e-4,
        x_min = -2,  # INTEGER
        x_max = 8    # INTEGER
    )

    # ===== 1. Value Iteration =====
    println("\n" * "=" ^ 70)
    println("Running Value Iteration...")
    @time V_vi, policy_vi, n_iter_vi, _ = value_iteration(mdp, max_iterations=1000, verbose=true)
    print_results(mdp, V_vi, policy_vi, "Value Iteration")

    # ===== 2. Policy Iteration (Iterative) =====
    println("\n" * "=" ^ 70)
    println("Running Policy Iteration (Iterative)...")
    @time V_pi, policy_pi, n_iter_pi = policy_iteration(mdp, max_iterations=100, verbose=true)
    print_results(mdp, V_pi, policy_pi, "Policy Iteration (Iterative)")

    # ===== 3. Policy Iteration (Monte Carlo) =====
    println("\n" * "=" ^ 70)
    println("Running Policy Iteration (Monte Carlo)...")
    @time V_mc, policy_mc, n_iter_mc = policy_iteration_mc(mdp,
        N = 50,
        n_episodes = 3000,
        max_iterations = 15,
        verbose = true
    )
    print_results(mdp, V_mc, policy_mc, "Policy Iteration (Monte Carlo)")

    # ===== 比较所有方法 =====
    println("\n" * "=" ^ 70)
    println("COMPARISON OF ALL METHODS (DISCRETE VERSION)")
    println("=" ^ 70)

    S_analytical = analytical_base_stock(mdp)

    println("\nBase-Stock Levels (Integer):")
    @printf("  Analytical:                S* = %d\n", S_analytical)
    @printf("  Value Iteration:           S* = %d (error: %d)\n",
            policy_vi[1], abs(policy_vi[1] - S_analytical))
    @printf("  Policy Iteration (Iter):   S* = %d (error: %d)\n",
            policy_pi[1], abs(policy_pi[1] - S_analytical))
    @printf("  Policy Iteration (MC):     S* = %d (error: %d)\n",
            policy_mc[1], abs(policy_mc[1] - S_analytical))

    println("\nIterations to converge:")
    @printf("  Value Iteration:           %d iterations\n", n_iter_vi)
    @printf("  Policy Iteration (Iter):   %d iterations\n", n_iter_pi)
    @printf("  Policy Iteration (MC):     %d iterations\n", n_iter_mc)

    # ===== 画图 =====
    println("\n" * "=" ^ 70)
    println("Generating plots...")

    p1 = plot_results(mdp, V_vi, policy_vi, "VI: ")
    p2 = plot_results(mdp, V_pi, policy_pi, "PI (Iter): ")
    p3 = plot_results(mdp, V_mc, policy_mc, "PI (MC): ")

    p = plot(p1, p2, p3, layout=(3, 1), size=(1000, 1000))
    display(p)

    println("\n" * "=" ^ 70)
    println("DISCRETE VERSION - All optimal actions are INTEGERS")
    println("No fractional values like 3.5 or 3.6!")
    println("=" ^ 70)

    return mdp, V_vi, policy_vi, V_pi, policy_pi, V_mc, policy_mc
end

# ============================================================================
# 运行
# ============================================================================

mdp, V_vi, policy_vi, V_pi, policy_pi, V_mc, policy_mc = main()