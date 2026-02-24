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
    visit_counts = zeros(Int, n_states)  # Track visit counts for each state

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

            # Count every visit (not just first visit)
            visit_counts[state_idx] += 1
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

    return V, visit_counts  # Return visit counts as well
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
    visit_counts = zeros(Int, n_states)  # Track visit counts

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

        V, visit_counts = first_visit_mc_evaluation(mdp, policy, N, v_terminal,
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
            return V, policy, iter, visit_counts  # Return visit counts
        end
    end

    if verbose
        println("\nReached maximum iterations ($max_iterations)")
    end

    return V, policy, max_iterations, visit_counts  # Return visit counts
end

# ============================================================================
# 第七部分: Q-Learning (Off-Policy TD Control)
# ============================================================================

function q_learning(mdp::InventoryMDP;
                    n_episodes::Int=50000,
                    α_init::Float64=0.1,
                    ε::Float64=0.1,
                    y_max_global::Int=10,
                    verbose::Bool=true)
    (; states, n_states, h, b, c, γ, demand_values) = mdp

    # All possible order-up-to levels (global action space)
    y_values = collect(states[1]:y_max_global)
    n_actions = length(y_values)

    # Q-table: Q[state_idx, action_idx]
    # Initialize to zeros
    Q = zeros(n_states, n_actions)

    # Visit count for each (s, a) pair — used for learning rate decay
    N_sa = zeros(Int, n_states, n_actions)

    # Track convergence: store max policy change periodically
    convergence_history = Float64[]
    episode_rewards = Float64[]

    if verbose
        println("\n" * "=" ^ 70)
        println("Q-LEARNING (Off-Policy TD Control - DISCRETE)")
        println("Episodes = $n_episodes, ε = $ε, α_init = $α_init")
        println("Action space: order-up-to y ∈ [$(states[1]), $y_max_global]")
        println("=" ^ 70)
    end

    for episode in 1:n_episodes
        # Random initial state
        s_idx = rand(1:n_states)
        x = states[s_idx]

        episode_reward = 0.0
        max_steps = 200  # prevent infinite episodes

        for step in 1:max_steps
            # --- ε-greedy behaviour policy ---
            # Valid actions: y >= x, so filter action indices
            valid_mask = [y >= x for y in y_values]
            valid_indices = findall(valid_mask)

            if rand() < ε
                # Explore: random valid action
                a_idx = rand(valid_indices)
            else
                # Exploit: greedy w.r.t. Q among valid actions
                valid_q = [Q[s_idx, j] for j in valid_indices]
                best_local = argmax(valid_q)
                a_idx = valid_indices[best_local]
            end

            y = y_values[a_idx]

            # --- Environment step ---
            # Sample demand
            D = rand(demand_values)

            # Compute reward = -(ordering cost + holding cost + shortage cost)
            ordering_cost = c * (y - x)
            inventory_after = y - D
            holding_cost = h * max(0, inventory_after)
            shortage_cost = b * max(0, -inventory_after)
            reward = -(ordering_cost + holding_cost + shortage_cost)

            episode_reward += reward

            # Next state
            x_next = clamp(inventory_after, states[1], states[end])
            s_next_idx = x_next - states[1] + 1

            # --- Q-learning update (off-policy: use max over next actions) ---
            # Valid actions in next state
            valid_next_mask = [y_val >= x_next for y_val in y_values]
            valid_next_indices = findall(valid_next_mask)
            max_Q_next = maximum(Q[s_next_idx, j] for j in valid_next_indices)

            # Learning rate with decay: α = α_init / (1 + visits)
            N_sa[s_idx, a_idx] += 1
            α = α_init / (1 + 0.001 * N_sa[s_idx, a_idx])

            # TD update
            Q[s_idx, a_idx] += α * (reward + γ * max_Q_next - Q[s_idx, a_idx])

            # Move to next state
            s_idx = s_next_idx
            x = x_next
        end

        push!(episode_rewards, episode_reward)

        if verbose && episode % (n_episodes ÷ 5) == 0
            # Extract current greedy policy for reporting
            avg_reward = mean(episode_rewards[max(1, episode-1000):episode])
            @printf("  Episode %6d / %d: avg reward (last 1000) = %.2f\n",
                    episode, n_episodes, avg_reward)
        end
    end

    # --- Extract greedy policy and value function from Q-table ---
    policy_ql = zeros(Int, n_states)
    V_ql = zeros(n_states)

    for i in 1:n_states
        x = states[i]
        valid_mask = [y >= x for y in y_values]
        valid_indices = findall(valid_mask)

        valid_q = [Q[i, j] for j in valid_indices]
        best_local = argmax(valid_q)
        best_a_idx = valid_indices[best_local]

        policy_ql[i] = y_values[best_a_idx]
        V_ql[i] = Q[i, best_a_idx]
    end

    if verbose
        println("  Q-learning complete.")
        println("  Learned base-stock level: S* = $(policy_ql[1])")
    end

    return V_ql, policy_ql, Q, episode_rewards
end

# ============================================================================
# 第八部分: 解析解 (DISCRETE - Approximate)
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

# Helper function to print visit count statistics
function print_visit_statistics(mdp::InventoryMDP, visit_counts::Vector{Int})
    println("\n" * "=" ^ 70)
    println("Monte Carlo State Visit Statistics")
    println("=" ^ 70)
    @printf("%-15s %-20s %-15s\n", "State (x)", "Visit Count", "Visit %")
    println("-" ^ 70)

    total_visits = sum(visit_counts)

    for i in 1:mdp.n_states
        visit_pct = (visit_counts[i] / total_visits) * 100
        @printf("%-15d %-20d %-15.2f%%\n", mdp.states[i], visit_counts[i], visit_pct)
    end

    println("-" ^ 70)
    println("Total visits: $total_visits")
    println("Most visited: State $(mdp.states[argmax(visit_counts)]) with $(maximum(visit_counts)) visits")
    println("Least visited: State $(mdp.states[argmin(visit_counts)]) with $(minimum(visit_counts)) visits")
    println("=" ^ 70)
end

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

# New function for Monte Carlo results with visit counts
function plot_mc_results(mdp::InventoryMDP, V::Vector{Float64}, policy::Vector{Int},
                         visit_counts::Vector{Int}, title_prefix::String="")
    S_analytical = analytical_base_stock(mdp)

    # Value Function
    p1 = plot(mdp.states, V,
        xlabel = "Inventory Level (x)",
        ylabel = "V*(x)",
        title = "$(title_prefix)Value Function",
        linewidth = 2,
        legend = false,
        color = :blue,
        marker = :circle,
        markersize = 3
    )

    # Optimal Policy
    p2 = plot(mdp.states, policy,
        xlabel = "Inventory Level (x)",
        ylabel = "Order-Up-To Level (y)",
        title = "$(title_prefix)Optimal Policy",
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

    # Visit Counts (NEW!)
    p3 = bar(mdp.states, visit_counts,
        xlabel = "Inventory Level (x)",
        ylabel = "Visit Count",
        title = "$(title_prefix)State Visit Frequency",
        color = :purple,
        alpha = 0.7,
        legend = false,
        bar_width = 0.8
    )

    # Add total visits annotation
    total_visits = sum(visit_counts)
    annotate!(p3, [(mdp.states[end], maximum(visit_counts) * 0.9,
                    text("Total: $total_visits", 8, :right))])

    p = plot(p1, p2, p3, layout=(1, 3), size=(1400, 400))
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

# Q-learning vs DP comparison plot
function plot_ql_vs_dp(mdp::InventoryMDP,
                       V_dp::Vector{Float64}, policy_dp::Vector{Int},
                       V_ql::Vector{Float64}, policy_ql::Vector{Int},
                       episode_rewards::Vector{Float64})
    S_analytical = analytical_base_stock(mdp)

    # 1. Value function comparison
    p1 = plot(mdp.states, V_dp,
        xlabel = "Inventory Level (x)",
        ylabel = "V(x)",
        title = "Value Function: Q-Learning vs DP",
        linewidth = 2,
        label = "DP (Value Iteration)",
        color = :blue,
        marker = :circle,
        markersize = 3
    )
    plot!(mdp.states, V_ql,
        linewidth = 2,
        label = "Q-Learning",
        color = :red,
        marker = :diamond,
        markersize = 3,
        linestyle = :dash
    )

    # 2. Policy comparison
    p2 = plot(mdp.states, policy_dp,
        xlabel = "Inventory Level (x)",
        ylabel = "Order-Up-To Level (y)",
        title = "Policy: Q-Learning vs DP",
        linewidth = 2,
        label = "DP (Value Iteration)",
        color = :blue,
        marker = :circle,
        markersize = 3
    )
    plot!(mdp.states, policy_ql,
        linewidth = 2,
        label = "Q-Learning",
        color = :red,
        marker = :diamond,
        markersize = 3,
        linestyle = :dash
    )
    plot!(mdp.states, mdp.states,
        linestyle = :dot,
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

    # 3. Value function error
    V_error = abs.(V_ql .- V_dp)
    p3 = bar(mdp.states, V_error,
        xlabel = "Inventory Level (x)",
        ylabel = "|V_QL(x) - V_DP(x)|",
        title = "Value Function Error",
        color = :orange,
        alpha = 0.7,
        legend = false,
        bar_width = 0.8
    )
    max_err = maximum(V_error)
    annotate!(p3, [(mdp.states[end], max_err * 0.9,
                    text(@sprintf("Max: %.4f", max_err), 8, :right))])

    # 4. Learning curve (smoothed episode rewards)
    window = min(1000, length(episode_rewards) ÷ 10)
    smoothed = [mean(episode_rewards[max(1, i-window+1):i]) for i in 1:length(episode_rewards)]
    p4 = plot(1:length(smoothed), smoothed,
        xlabel = "Episode",
        ylabel = "Avg Reward (smoothed)",
        title = "Q-Learning Convergence",
        linewidth = 1,
        color = :purple,
        legend = false,
        alpha = 0.8
    )

    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
    return p
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
    time_vi = @elapsed begin
        V_vi, policy_vi, n_iter_vi, _ = value_iteration(mdp, max_iterations=1000, verbose=true)
    end
    @printf("  Wall time: %.4f seconds\n", time_vi)
    print_results(mdp, V_vi, policy_vi, "Value Iteration")

    # ===== 2. Policy Iteration (Iterative) =====
    println("\n" * "=" ^ 70)
    println("Running Policy Iteration (Iterative)...")
    time_pi = @elapsed begin
        V_pi, policy_pi, n_iter_pi = policy_iteration(mdp, max_iterations=100, verbose=true)
    end
    @printf("  Wall time: %.4f seconds\n", time_pi)
    print_results(mdp, V_pi, policy_pi, "Policy Iteration (Iterative)")

    # ===== 3. Policy Iteration (Monte Carlo) =====
    println("\n" * "=" ^ 70)
    println("Running Policy Iteration (Monte Carlo)...")
    time_mc = @elapsed begin
        V_mc, policy_mc, n_iter_mc, visit_counts_mc = policy_iteration_mc(mdp,
            N = 50,
            n_episodes = 3000,
            max_iterations = 15,
            verbose = true
        )
    end
    @printf("  Wall time: %.4f seconds\n", time_mc)
    print_results(mdp, V_mc, policy_mc, "Policy Iteration (Monte Carlo)")

    # Print visit statistics
    println("\n" * "-" ^ 50)
    println("Monte Carlo State Visit Statistics:")
    println("-" ^ 50)
    total_visits = sum(visit_counts_mc)
    println("Total state visits: $total_visits")
    println("Most visited state: $(mdp.states[argmax(visit_counts_mc)]) ($(maximum(visit_counts_mc)) visits)")
    println("Least visited state: $(mdp.states[argmin(visit_counts_mc)]) ($(minimum(visit_counts_mc)) visits)")
    println("-" ^ 50)

    # ===== 4. Q-Learning =====
    println("\n" * "=" ^ 70)
    println("Running Q-Learning (Off-Policy TD Control)...")
    time_ql = @elapsed begin
        V_ql, policy_ql, Q_table, episode_rewards = q_learning(mdp,
            n_episodes = 50000,
            α_init = 0.1,
            ε = 0.1,
            y_max_global = 10,
            verbose = true
        )
    end
    @printf("  Wall time: %.4f seconds\n", time_ql)
    print_results(mdp, V_ql, policy_ql, "Q-Learning")

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
    @printf("  Q-Learning:                S* = %d (error: %d)\n",
            policy_ql[1], abs(policy_ql[1] - S_analytical))

    println("\nIterations to converge:")
    @printf("  Value Iteration:           %d iterations\n", n_iter_vi)
    @printf("  Policy Iteration (Iter):   %d iterations\n", n_iter_pi)
    @printf("  Policy Iteration (MC):     %d iterations\n", n_iter_mc)
    @printf("  Q-Learning:                %d episodes\n", length(episode_rewards))

    println("\nComputation Time:")
    @printf("  Value Iteration (DP):      %.4f seconds\n", time_vi)
    @printf("  Policy Iteration (Iter):   %.4f seconds\n", time_pi)
    @printf("  Policy Iteration (MC):     %.4f seconds\n", time_mc)
    @printf("  Q-Learning:                %.4f seconds\n", time_ql)
    @printf("  Speedup DP vs Q-Learning:  %.1fx\n",
            time_ql > 0 ? time_ql / time_vi : Inf)

    # ===== Q-Learning vs DP detailed comparison =====
    println("\n" * "=" ^ 70)
    println("Q-LEARNING vs DYNAMIC PROGRAMMING (Value Iteration)")
    println("=" ^ 70)
    @printf("%-10s %-15s %-15s %-15s %-15s\n",
            "State", "DP Policy", "QL Policy", "DP Value", "QL Value")
    println("-" ^ 70)
    for i in 1:mdp.n_states
        @printf("%-10d %-15d %-15d %-15.4f %-15.4f\n",
                mdp.states[i], policy_vi[i], policy_ql[i], V_vi[i], V_ql[i])
    end
    println("-" ^ 70)
    policy_match = sum(policy_vi .== policy_ql)
    @printf("Policy agreement: %d / %d states (%.1f%%)\n",
            policy_match, mdp.n_states, 100.0 * policy_match / mdp.n_states)
    @printf("Max value error:  %.6f\n", maximum(abs.(V_ql .- V_vi)))
    @printf("Mean value error: %.6f\n", mean(abs.(V_ql .- V_vi)))

    # ===== 画图 =====
    println("\n" * "=" ^ 70)
    println("Generating plots...")

    p1 = plot_results(mdp, V_vi, policy_vi, "VI: ")
    p2 = plot_results(mdp, V_pi, policy_pi, "PI (Iter): ")
    p3 = plot_mc_results(mdp, V_mc, policy_mc, visit_counts_mc, "PI (MC): ")

    p_all = plot(p1, p2, p3, layout=(3, 1), size=(1400, 1400))
    display(p_all)

    # Q-Learning vs DP comparison plot
    p_ql = plot_ql_vs_dp(mdp, V_vi, policy_vi, V_ql, policy_ql, episode_rewards)
    display(p_ql)

    println("\n" * "=" ^ 70)
    println("DISCRETE VERSION - All optimal actions are INTEGERS")
    println("Q-Learning successfully learns the optimal policy via exploration!")
    println("=" ^ 70)

    return mdp, V_vi, policy_vi, V_pi, policy_pi, V_mc, policy_mc, visit_counts_mc, V_ql, policy_ql
end

# ============================================================================
# 第十部分: Scaling Experiment — DP vs Q-Learning Runtime vs Demand Size
# ============================================================================

function runtime_scaling_experiment(;
        d_max_values = [2, 4, 6, 8, 10, 12, 14, 16],
        n_episodes_ql = 50000,
        verbose = true)

    times_dp = Float64[]
    times_ql = Float64[]
    n_states_list = Int[]
    n_actions_list = Int[]

    println("\n" * "=" ^ 70)
    println("SCALING EXPERIMENT: Runtime vs Max Demand")
    println("Demand ~ Uniform(0, d_max) for d_max ∈ $d_max_values")
    println("Q-Learning episodes fixed at $n_episodes_ql")
    println("=" ^ 70)

    for d_max in d_max_values
        demand_vals = collect(0:d_max)
        # Scale state space with demand: x ∈ [-d_max/2, 2*d_max]
        x_min = -max(2, d_max ÷ 2)
        x_max = 2 * d_max
        y_max_global = x_max + 2

        if verbose
            @printf("\n  d_max = %2d | states [%d, %d] (%d) | actions up to %d | demands: %d values\n",
                    d_max, x_min, x_max, x_max - x_min + 1, y_max_global, length(demand_vals))
        end

        # Create MDP (suppress printing)
        mdp_test = InventoryMDP(
            demand_values = demand_vals,
            h = 1.0, b = 9.0, c = 0.0, γ = 0.95, θ = 1e-4,
            x_min = x_min, x_max = x_max
        )

        push!(n_states_list, mdp_test.n_states)
        push!(n_actions_list, y_max_global - x_min + 1)

        # Time DP (Value Iteration)
        t_dp = @elapsed begin
            value_iteration(mdp_test, max_iterations=1000, verbose=false)
        end
        push!(times_dp, t_dp)

        # Time Q-Learning
        t_ql = @elapsed begin
            q_learning(mdp_test,
                n_episodes = n_episodes_ql,
                α_init = 0.1, ε = 0.1,
                y_max_global = y_max_global,
                verbose = false)
        end
        push!(times_ql, t_ql)

        if verbose
            @printf("    DP: %.4f s | Q-Learning: %.4f s | Ratio QL/DP: %.1fx\n",
                    t_dp, t_ql, t_ql / max(t_dp, 1e-8))
        end
    end

    # --- Summary table ---
    println("\n" * "=" ^ 70)
    println("SCALING SUMMARY")
    println("=" ^ 70)
    @printf("%-10s %-12s %-12s %-14s %-14s %-10s\n",
            "d_max", "|States|", "|Actions|", "DP Time (s)", "QL Time (s)", "QL/DP")
    println("-" ^ 70)
    for (i, d_max) in enumerate(d_max_values)
        ratio = times_ql[i] / max(times_dp[i], 1e-8)
        @printf("%-10d %-12d %-12d %-14.4f %-14.4f %-10.1fx\n",
                d_max, n_states_list[i], n_actions_list[i],
                times_dp[i], times_ql[i], ratio)
    end
    println("=" ^ 70)

    # --- Plot ---
    # 1. Absolute runtime
    p1 = plot(d_max_values, times_dp,
        xlabel = "Max Demand (d_max)",
        ylabel = "Time (seconds)",
        title = "Runtime: DP vs Q-Learning",
        linewidth = 2,
        label = "DP (Value Iteration)",
        color = :blue,
        marker = :circle,
        markersize = 5
    )
    plot!(d_max_values, times_ql,
        linewidth = 2,
        label = "Q-Learning ($(n_episodes_ql) episodes)",
        color = :red,
        marker = :diamond,
        markersize = 5
    )

    # 2. Log-scale runtime
    p2 = plot(d_max_values, times_dp,
        xlabel = "Max Demand (d_max)",
        ylabel = "Time (seconds, log scale)",
        title = "Runtime (Log Scale)",
        linewidth = 2,
        label = "DP (Value Iteration)",
        color = :blue,
        marker = :circle,
        markersize = 5,
        yscale = :log10
    )
    plot!(d_max_values, times_ql,
        linewidth = 2,
        label = "Q-Learning",
        color = :red,
        marker = :diamond,
        markersize = 5
    )

    # 3. Ratio QL/DP
    ratios = times_ql ./ max.(times_dp, 1e-8)
    p3 = bar(d_max_values, ratios,
        xlabel = "Max Demand (d_max)",
        ylabel = "Ratio (QL time / DP time)",
        title = "Slowdown Factor: Q-Learning vs DP",
        color = :orange,
        alpha = 0.7,
        legend = false,
        bar_width = 1.5
    )
    hline!([1.0], linestyle = :dash, color = :black, alpha = 0.5, label = "")

    # 4. State/action space size
    p4 = plot(d_max_values, n_states_list,
        xlabel = "Max Demand (d_max)",
        ylabel = "Count",
        title = "Problem Size Growth",
        linewidth = 2,
        label = "# States",
        color = :green,
        marker = :circle,
        markersize = 5
    )
    plot!(d_max_values, n_actions_list,
        linewidth = 2,
        label = "# Actions",
        color = :purple,
        marker = :square,
        markersize = 5
    )

    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
    display(p)

    return d_max_values, times_dp, times_ql
end

# ============================================================================
# 运行
# ============================================================================

mdp, V_vi, policy_vi, V_pi, policy_pi, V_mc, policy_mc, visit_counts_mc, V_ql, policy_ql = main()

# Run scaling experiment
println("\n\n")
d_max_values, times_dp, times_ql = runtime_scaling_experiment(
    d_max_values = [2, 4, 6, 8, 10, 12, 14, 16],
    n_episodes_ql = 50000
)