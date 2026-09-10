import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class BernoulliBandit:
    """Multi-armed bandit with Bernoulli rewards"""
    def __init__(self, probabilities):
        self.probabilities = np.array(probabilities)
        self.n_arms = len(probabilities)
        self.optimal_value = np.max(probabilities)

    def pull(self, arm):
        """Pull an arm and get a Bernoulli reward (0 or 1)"""
        return np.random.binomial(1, self.probabilities[arm])


class EpsilonGreedy:
    """Epsilon-greedy algorithm"""
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        """Select an arm using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm so far
            return np.argmax(self.values)

    def update(self, arm, reward):
        """Update estimates after observing a reward"""
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental average
        self.values[arm] += (reward - self.values[arm]) / n


class UCB:
    """Upper Confidence Bound algorithm"""
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c  # Exploration parameter
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0  # Total time steps

    def select_arm(self):
        """Select an arm using UCB strategy"""
        self.t += 1

        # Pull each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # UCB formula: value + c * sqrt(ln(t) / n)
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update estimates after observing a reward"""
        self.counts[arm] += 1
        n = self.counts[arm]
        # Incremental average
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSampling:
    """Thompson Sampling algorithm with Beta-Bernoulli conjugate prior"""
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta distribution parameters: alpha (successes), beta (failures)
        # Start with Beta(1,1) = Uniform(0,1) prior
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self):
        """Select an arm by sampling from each arm's posterior distribution"""
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)
        # Return arm with highest sample
        return np.argmax(samples)

    def update(self, arm, reward):
        """Update Beta distribution parameters based on observed reward"""
        # Bernoulli reward: 1 = success, 0 = failure
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)


def run_experiment(algorithm, bandit, n_steps):
    """Run a bandit algorithm for n_steps and track all metrics"""
    cumulative_regrets = np.zeros(n_steps)
    instantaneous_regrets = np.zeros(n_steps)
    cumulative_rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    cumulative_regret = 0
    cumulative_reward = 0
    optimal_arm = np.argmax(bandit.probabilities)

    for t in range(n_steps):
        # Select and pull arm
        arm = algorithm.select_arm()
        reward = bandit.pull(arm)

        # Update algorithm
        algorithm.update(arm, reward)

        # Calculate instantaneous regret (optimal - actual expected reward)
        inst_regret = bandit.optimal_value - bandit.probabilities[arm]
        cumulative_regret += inst_regret
        cumulative_reward += reward

        # Track metrics
        instantaneous_regrets[t] = inst_regret
        cumulative_regrets[t] = cumulative_regret
        cumulative_rewards[t] = cumulative_reward
        optimal_actions[t] = 1 if arm == optimal_arm else 0

    return cumulative_regrets, instantaneous_regrets, cumulative_rewards, optimal_actions


def compare_algorithms(probabilities, n_steps=1000, n_runs=100):
    """Compare epsilon-greedy, UCB, and Thompson Sampling algorithms"""

    # Store results for multiple runs
    eg_cum_regrets = np.zeros((n_runs, n_steps))
    eg_inst_regrets = np.zeros((n_runs, n_steps))
    eg_cum_rewards = np.zeros((n_runs, n_steps))
    eg_optimal_actions = np.zeros((n_runs, n_steps))

    ucb_cum_regrets = np.zeros((n_runs, n_steps))
    ucb_inst_regrets = np.zeros((n_runs, n_steps))
    ucb_cum_rewards = np.zeros((n_runs, n_steps))
    ucb_optimal_actions = np.zeros((n_runs, n_steps))

    ts_cum_regrets = np.zeros((n_runs, n_steps))
    ts_inst_regrets = np.zeros((n_runs, n_steps))
    ts_cum_rewards = np.zeros((n_runs, n_steps))
    ts_optimal_actions = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        # Create bandit
        bandit = BernoulliBandit(probabilities)

        # Run epsilon-greedy
        eg = EpsilonGreedy(n_arms=len(probabilities), epsilon=0.1)
        eg_cum_regrets[run], eg_inst_regrets[run], eg_cum_rewards[run], eg_optimal_actions[run] = \
            run_experiment(eg, bandit, n_steps)

        # Run UCB
        ucb = UCB(n_arms=len(probabilities), c=2.0)
        ucb_cum_regrets[run], ucb_inst_regrets[run], ucb_cum_rewards[run], ucb_optimal_actions[run] = \
            run_experiment(ucb, bandit, n_steps)

        # Run Thompson Sampling
        ts = ThompsonSampling(n_arms=len(probabilities))
        ts_cum_regrets[run], ts_inst_regrets[run], ts_cum_rewards[run], ts_optimal_actions[run] = \
            run_experiment(ts, bandit, n_steps)

    # Calculate statistics
    results = {
        'eg': {
            'cum_regret_mean': np.mean(eg_cum_regrets, axis=0),
            'cum_regret_std': np.std(eg_cum_regrets, axis=0),
            'inst_regret_mean': np.mean(eg_inst_regrets, axis=0),
            'inst_regret_std': np.std(eg_inst_regrets, axis=0),
            'cum_reward_mean': np.mean(eg_cum_rewards, axis=0),
            'cum_reward_std': np.std(eg_cum_rewards, axis=0),
            'optimal_pct': np.mean(eg_optimal_actions, axis=0) * 100
        },
        'ucb': {
            'cum_regret_mean': np.mean(ucb_cum_regrets, axis=0),
            'cum_regret_std': np.std(ucb_cum_regrets, axis=0),
            'inst_regret_mean': np.mean(ucb_inst_regrets, axis=0),
            'inst_regret_std': np.std(ucb_inst_regrets, axis=0),
            'cum_reward_mean': np.mean(ucb_cum_rewards, axis=0),
            'cum_reward_std': np.std(ucb_cum_rewards, axis=0),
            'optimal_pct': np.mean(ucb_optimal_actions, axis=0) * 100
        },
        'ts': {
            'cum_regret_mean': np.mean(ts_cum_regrets, axis=0),
            'cum_regret_std': np.std(ts_cum_regrets, axis=0),
            'inst_regret_mean': np.mean(ts_inst_regrets, axis=0),
            'inst_regret_std': np.std(ts_inst_regrets, axis=0),
            'cum_reward_mean': np.mean(ts_cum_rewards, axis=0),
            'cum_reward_std': np.std(ts_cum_rewards, axis=0),
            'optimal_pct': np.mean(ts_optimal_actions, axis=0) * 100
        }
    }

    return results


def plot_all_metrics(results, probabilities):
    """Plot comprehensive comparison of all metrics"""
    n_steps = len(results['eg']['cum_regret_mean'])
    steps = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bandit Algorithm Comparison\nArm Probabilities: {probabilities}',
                 fontsize=14, fontweight='bold')

    # ---- Subplot 1: Cumulative Regret ----
    ax = axes[0, 0]
    ax.plot(steps, results['eg']['cum_regret_mean'], label='ε-Greedy (ε=0.1)', linewidth=2)
    ax.fill_between(steps,
                     results['eg']['cum_regret_mean'] - results['eg']['cum_regret_std'],
                     results['eg']['cum_regret_mean'] + results['eg']['cum_regret_std'],
                     alpha=0.2)

    ax.plot(steps, results['ucb']['cum_regret_mean'], label='UCB (c=2.0)', linewidth=2)
    ax.fill_between(steps,
                     results['ucb']['cum_regret_mean'] - results['ucb']['cum_regret_std'],
                     results['ucb']['cum_regret_mean'] + results['ucb']['cum_regret_std'],
                     alpha=0.2)

    ax.plot(steps, results['ts']['cum_regret_mean'], label='Thompson Sampling', linewidth=2)
    ax.fill_between(steps,
                     results['ts']['cum_regret_mean'] - results['ts']['cum_regret_std'],
                     results['ts']['cum_regret_mean'] + results['ts']['cum_regret_std'],
                     alpha=0.2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Cumulative Regret', fontsize=10)
    ax.set_title('Cumulative Regret', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 2: Cumulative Reward ----
    ax = axes[0, 1]
    ax.plot(steps, results['eg']['cum_reward_mean'], label='ε-Greedy (ε=0.1)', linewidth=2)
    ax.fill_between(steps,
                     results['eg']['cum_reward_mean'] - results['eg']['cum_reward_std'],
                     results['eg']['cum_reward_mean'] + results['eg']['cum_reward_std'],
                     alpha=0.2)

    ax.plot(steps, results['ucb']['cum_reward_mean'], label='UCB (c=2.0)', linewidth=2)
    ax.fill_between(steps,
                     results['ucb']['cum_reward_mean'] - results['ucb']['cum_reward_std'],
                     results['ucb']['cum_reward_mean'] + results['ucb']['cum_reward_std'],
                     alpha=0.2)

    ax.plot(steps, results['ts']['cum_reward_mean'], label='Thompson Sampling', linewidth=2)
    ax.fill_between(steps,
                     results['ts']['cum_reward_mean'] - results['ts']['cum_reward_std'],
                     results['ts']['cum_reward_mean'] + results['ts']['cum_reward_std'],
                     alpha=0.2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Cumulative Reward', fontsize=10)
    ax.set_title('Cumulative Reward', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 3: Instantaneous Regret (smoothed) ----
    ax = axes[1, 0]
    # Apply moving average for smoother visualization
    window = 50
    eg_inst_smooth = np.convolve(results['eg']['inst_regret_mean'],
                                  np.ones(window)/window, mode='valid')
    ucb_inst_smooth = np.convolve(results['ucb']['inst_regret_mean'],
                                   np.ones(window)/window, mode='valid')
    ts_inst_smooth = np.convolve(results['ts']['inst_regret_mean'],
                                  np.ones(window)/window, mode='valid')
    steps_smooth = steps[:len(eg_inst_smooth)]

    ax.plot(steps_smooth, eg_inst_smooth, label='ε-Greedy (ε=0.1)', linewidth=2)
    ax.plot(steps_smooth, ucb_inst_smooth, label='UCB (c=2.0)', linewidth=2)
    ax.plot(steps_smooth, ts_inst_smooth, label='Thompson Sampling', linewidth=2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Instantaneous Regret', fontsize=10)
    ax.set_title(f'Instantaneous Regret (smoothed, window={window})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Subplot 4: % Optimal Action ----
    ax = axes[1, 1]
    ax.plot(steps, results['eg']['optimal_pct'], label='ε-Greedy (ε=0.1)', linewidth=2)
    ax.plot(steps, results['ucb']['optimal_pct'], label='UCB (c=2.0)', linewidth=2)
    ax.plot(steps, results['ts']['optimal_pct'], label='Thompson Sampling', linewidth=2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('% Optimal Action', fontsize=10)
    ax.set_title('Percentage of Optimal Actions', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bandit_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'bandit_comparison.png'")
    # plt.show()  # Commented out - using non-interactive backend


if __name__ == "__main__":
    # Define arm probabilities (Bernoulli parameters)
    # Example: 5 arms with different success probabilities
    probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("Multi-Armed Bandit Comparison")
    print("=" * 50)
    print(f"Arm probabilities: {probabilities}")
    print(f"Optimal arm: {np.argmax(probabilities)} (p={np.max(probabilities)})")
    print("=" * 50)

    # Run comparison
    print("\nRunning experiments...")
    results = compare_algorithms(
        probabilities,
        n_steps=5000,
        n_runs=100
    )

    # Print final results
    print(f"\nFinal Results (after 5000 steps):")
    print("-" * 50)
    print("Cumulative Regret:")
    print(f"  ε-Greedy:          {results['eg']['cum_regret_mean'][-1]:.2f} ± {results['eg']['cum_regret_std'][-1]:.2f}")
    print(f"  UCB:               {results['ucb']['cum_regret_mean'][-1]:.2f} ± {results['ucb']['cum_regret_std'][-1]:.2f}")
    print(f"  Thompson Sampling: {results['ts']['cum_regret_mean'][-1]:.2f} ± {results['ts']['cum_regret_std'][-1]:.2f}")

    print("\nCumulative Reward:")
    print(f"  ε-Greedy:          {results['eg']['cum_reward_mean'][-1]:.2f} ± {results['eg']['cum_reward_std'][-1]:.2f}")
    print(f"  UCB:               {results['ucb']['cum_reward_mean'][-1]:.2f} ± {results['ucb']['cum_reward_std'][-1]:.2f}")
    print(f"  Thompson Sampling: {results['ts']['cum_reward_mean'][-1]:.2f} ± {results['ts']['cum_reward_std'][-1]:.2f}")

    print("\n% Optimal Action:")
    print(f"  ε-Greedy:          {results['eg']['optimal_pct'][-1]:.2f}%")
    print(f"  UCB:               {results['ucb']['optimal_pct'][-1]:.2f}%")
    print(f"  Thompson Sampling: {results['ts']['optimal_pct'][-1]:.2f}%")

    # Plot results
    plot_all_metrics(results, probabilities)
