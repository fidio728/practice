import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class LinearContextualBandit:
    """Contextual bandit where reward = context^T * theta_a + noise"""
    def __init__(self, n_arms, context_dim):
        self.n_arms = n_arms
        self.context_dim = context_dim

        # True parameters for each arm (unknown to algorithms)
        # Each arm has different weights for context features
        self.theta = np.random.randn(n_arms, context_dim)
        self.optimal_arm_history = []

    def get_context(self):
        """Generate a random context vector"""
        return np.random.randn(self.context_dim)

    def get_reward(self, context, arm):
        """Get reward for pulling arm given context"""
        # Expected reward = context^T * theta_arm
        expected_reward = np.dot(context, self.theta[arm])
        # Add Gaussian noise
        reward = expected_reward + np.random.randn() * 0.1
        return reward

    def get_optimal_arm(self, context):
        """Return the optimal arm for given context"""
        expected_rewards = np.dot(self.theta, context)
        return np.argmax(expected_rewards)

    def get_optimal_reward(self, context):
        """Return the optimal expected reward for given context"""
        expected_rewards = np.dot(self.theta, context)
        return np.max(expected_rewards)


class EpsilonGreedyContextual:
    """ε-greedy with linear model for each arm"""
    def __init__(self, n_arms, context_dim, epsilon=0.1):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.epsilon = epsilon

        # Linear model: reward = context^T * theta for each arm
        self.theta = np.zeros((n_arms, context_dim))

        # For online linear regression (using Ridge regression)
        self.A = [np.eye(context_dim) for _ in range(n_arms)]  # Design matrix
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]  # Target vector

    def select_arm(self, context):
        """Select arm using ε-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: arm with highest predicted reward
            predicted_rewards = [np.dot(context, self.theta[a]) for a in range(self.n_arms)]
            return np.argmax(predicted_rewards)

    def update(self, context, arm, reward):
        """Update linear model for selected arm"""
        # Ridge regression update
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Solve for theta: theta = A^{-1} * b
        try:
            self.theta[arm] = np.linalg.solve(self.A[arm], self.b[arm])
        except np.linalg.LinAlgError:
            pass  # Keep previous theta if singular


class LinUCB:
    """Linear Upper Confidence Bound algorithm"""
    def __init__(self, n_arms, context_dim, alpha=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha  # Exploration parameter

        # Linear model parameters for each arm
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context):
        """Select arm using UCB policy"""
        ucb_values = []

        for a in range(self.n_arms):
            # Compute theta_a
            try:
                self.theta[a] = np.linalg.solve(self.A[a], self.b[a])
            except np.linalg.LinAlgError:
                self.theta[a] = np.zeros(self.context_dim)

            # Compute UCB: theta^T * x + alpha * sqrt(x^T * A^{-1} * x)
            predicted_reward = np.dot(self.theta[a], context)

            try:
                A_inv = np.linalg.inv(self.A[a])
                uncertainty = self.alpha * np.sqrt(np.dot(context, np.dot(A_inv, context)))
            except np.linalg.LinAlgError:
                uncertainty = self.alpha

            ucb_values.append(predicted_reward + uncertainty)

        return np.argmax(ucb_values)

    def update(self, context, arm, reward):
        """Update model for selected arm"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


class ThompsonSamplingContextual:
    """Thompson Sampling with Bayesian Linear Regression"""
    def __init__(self, n_arms, context_dim, v=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.v = v  # Noise variance (assumed known)

        # Bayesian linear regression parameters for each arm
        # Prior: N(mu_0, v * B_0^{-1})
        self.B = [np.eye(context_dim) for _ in range(n_arms)]
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context):
        """Select arm by sampling from posterior"""
        sampled_rewards = []

        for a in range(self.n_arms):
            # Update mu from B and f
            try:
                self.mu[a] = np.linalg.solve(self.B[a], self.f[a])
                B_inv = np.linalg.inv(self.B[a])
            except np.linalg.LinAlgError:
                self.mu[a] = np.zeros(self.context_dim)
                B_inv = np.eye(self.context_dim)

            # Sample theta from posterior: N(mu_a, v * B_a^{-1})
            theta_sample = np.random.multivariate_normal(
                self.mu[a],
                self.v * B_inv
            )

            # Compute expected reward for this sample
            sampled_reward = np.dot(theta_sample, context)
            sampled_rewards.append(sampled_reward)

        return np.argmax(sampled_rewards)

    def update(self, context, arm, reward):
        """Update posterior for selected arm"""
        self.B[arm] += np.outer(context, context) / self.v
        self.f[arm] += reward * context / self.v


def run_contextual_experiment(algorithm, bandit, n_steps):
    """Run contextual bandit experiment"""
    cumulative_regrets = np.zeros(n_steps)
    instantaneous_regrets = np.zeros(n_steps)
    cumulative_rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    cumulative_regret = 0
    cumulative_reward = 0

    for t in range(n_steps):
        # Get context
        context = bandit.get_context()

        # Select arm
        arm = algorithm.select_arm(context)

        # Get reward
        reward = bandit.get_reward(context, arm)

        # Update algorithm
        algorithm.update(context, arm, reward)

        # Calculate regret
        optimal_reward = bandit.get_optimal_reward(context)
        optimal_arm = bandit.get_optimal_arm(context)

        expected_reward = np.dot(context, bandit.theta[arm])
        inst_regret = optimal_reward - expected_reward

        cumulative_regret += inst_regret
        cumulative_reward += reward

        # Track metrics
        instantaneous_regrets[t] = inst_regret
        cumulative_regrets[t] = cumulative_regret
        cumulative_rewards[t] = cumulative_reward
        optimal_actions[t] = 1 if arm == optimal_arm else 0

    return cumulative_regrets, instantaneous_regrets, cumulative_rewards, optimal_actions


def compare_contextual_algorithms(n_arms=5, context_dim=10, n_steps=5000, n_runs=50):
    """Compare contextual bandit algorithms"""

    # Store results
    eg_cum_regrets = np.zeros((n_runs, n_steps))
    eg_inst_regrets = np.zeros((n_runs, n_steps))
    eg_cum_rewards = np.zeros((n_runs, n_steps))
    eg_optimal_actions = np.zeros((n_runs, n_steps))

    linucb_cum_regrets = np.zeros((n_runs, n_steps))
    linucb_inst_regrets = np.zeros((n_runs, n_steps))
    linucb_cum_rewards = np.zeros((n_runs, n_steps))
    linucb_optimal_actions = np.zeros((n_runs, n_steps))

    ts_cum_regrets = np.zeros((n_runs, n_steps))
    ts_inst_regrets = np.zeros((n_runs, n_steps))
    ts_cum_rewards = np.zeros((n_runs, n_steps))
    ts_optimal_actions = np.zeros((n_runs, n_steps))

    for run in range(n_runs):
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        # Create bandit
        bandit = LinearContextualBandit(n_arms, context_dim)

        # ε-greedy
        eg = EpsilonGreedyContextual(n_arms, context_dim, epsilon=0.1)
        eg_cum_regrets[run], eg_inst_regrets[run], eg_cum_rewards[run], eg_optimal_actions[run] = \
            run_contextual_experiment(eg, bandit, n_steps)

        # LinUCB
        linucb = LinUCB(n_arms, context_dim, alpha=1.0)
        linucb_cum_regrets[run], linucb_inst_regrets[run], linucb_cum_rewards[run], linucb_optimal_actions[run] = \
            run_contextual_experiment(linucb, bandit, n_steps)

        # Thompson Sampling
        ts = ThompsonSamplingContextual(n_arms, context_dim, v=1.0)
        ts_cum_regrets[run], ts_inst_regrets[run], ts_cum_rewards[run], ts_optimal_actions[run] = \
            run_contextual_experiment(ts, bandit, n_steps)

    # Calculate statistics
    results = {
        'eg': {
            'cum_regret_mean': np.mean(eg_cum_regrets, axis=0),
            'cum_regret_std': np.std(eg_cum_regrets, axis=0),
            'inst_regret_mean': np.mean(eg_inst_regrets, axis=0),
            'cum_reward_mean': np.mean(eg_cum_rewards, axis=0),
            'optimal_pct': np.mean(eg_optimal_actions, axis=0) * 100
        },
        'linucb': {
            'cum_regret_mean': np.mean(linucb_cum_regrets, axis=0),
            'cum_regret_std': np.std(linucb_cum_regrets, axis=0),
            'inst_regret_mean': np.mean(linucb_inst_regrets, axis=0),
            'cum_reward_mean': np.mean(linucb_cum_rewards, axis=0),
            'optimal_pct': np.mean(linucb_optimal_actions, axis=0) * 100
        },
        'ts': {
            'cum_regret_mean': np.mean(ts_cum_regrets, axis=0),
            'cum_regret_std': np.std(ts_cum_regrets, axis=0),
            'inst_regret_mean': np.mean(ts_inst_regrets, axis=0),
            'cum_reward_mean': np.mean(ts_cum_rewards, axis=0),
            'optimal_pct': np.mean(ts_optimal_actions, axis=0) * 100
        }
    }

    return results


def plot_contextual_results(results, n_arms, context_dim):
    """Plot comparison results"""
    n_steps = len(results['eg']['cum_regret_mean'])
    steps = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Contextual Bandit Comparison\n{n_arms} arms, {context_dim}-dimensional context',
                 fontsize=14, fontweight='bold')

    # Cumulative Regret
    ax = axes[0, 0]
    ax.plot(steps, results['eg']['cum_regret_mean'], label='ε-Greedy (ε=0.1)', linewidth=2)
    ax.fill_between(steps,
                     results['eg']['cum_regret_mean'] - results['eg']['cum_regret_std'],
                     results['eg']['cum_regret_mean'] + results['eg']['cum_regret_std'],
                     alpha=0.2)

    ax.plot(steps, results['linucb']['cum_regret_mean'], label='LinUCB (α=1.0)', linewidth=2)
    ax.fill_between(steps,
                     results['linucb']['cum_regret_mean'] - results['linucb']['cum_regret_std'],
                     results['linucb']['cum_regret_mean'] + results['linucb']['cum_regret_std'],
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

    # Cumulative Reward
    ax = axes[0, 1]
    ax.plot(steps, results['eg']['cum_reward_mean'], label='ε-Greedy', linewidth=2)
    ax.plot(steps, results['linucb']['cum_reward_mean'], label='LinUCB', linewidth=2)
    ax.plot(steps, results['ts']['cum_reward_mean'], label='Thompson Sampling', linewidth=2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Cumulative Reward', fontsize=10)
    ax.set_title('Cumulative Reward', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Instantaneous Regret (smoothed)
    ax = axes[1, 0]
    window = 50
    eg_smooth = np.convolve(results['eg']['inst_regret_mean'],
                            np.ones(window)/window, mode='valid')
    linucb_smooth = np.convolve(results['linucb']['inst_regret_mean'],
                                np.ones(window)/window, mode='valid')
    ts_smooth = np.convolve(results['ts']['inst_regret_mean'],
                            np.ones(window)/window, mode='valid')
    steps_smooth = steps[:len(eg_smooth)]

    ax.plot(steps_smooth, eg_smooth, label='ε-Greedy', linewidth=2)
    ax.plot(steps_smooth, linucb_smooth, label='LinUCB', linewidth=2)
    ax.plot(steps_smooth, ts_smooth, label='Thompson Sampling', linewidth=2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('Instantaneous Regret', fontsize=10)
    ax.set_title(f'Instantaneous Regret (smoothed, window={window})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # % Optimal Action
    ax = axes[1, 1]
    ax.plot(steps, results['eg']['optimal_pct'], label='ε-Greedy', linewidth=2)
    ax.plot(steps, results['linucb']['optimal_pct'], label='LinUCB', linewidth=2)
    ax.plot(steps, results['ts']['optimal_pct'], label='Thompson Sampling', linewidth=2)

    ax.set_xlabel('Time Steps', fontsize=10)
    ax.set_ylabel('% Optimal Action', fontsize=10)
    ax.set_title('Percentage of Optimal Actions', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('contextual_bandit_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'contextual_bandit_comparison.png'")


if __name__ == "__main__":
    # Settings
    n_arms = 5
    context_dim = 10
    n_steps = 5000
    n_runs = 50

    print("Contextual Bandit Comparison")
    print("=" * 60)
    print(f"Number of arms: {n_arms}")
    print(f"Context dimension: {context_dim}")
    print(f"Time steps: {n_steps}")
    print(f"Number of runs: {n_runs}")
    print("=" * 60)

    # Run comparison
    print("\nRunning experiments...")
    results = compare_contextual_algorithms(n_arms, context_dim, n_steps, n_runs)

    # Print final results
    print(f"\nFinal Results (after {n_steps} steps):")
    print("-" * 60)
    print("Cumulative Regret:")
    print(f"  ε-Greedy:          {results['eg']['cum_regret_mean'][-1]:.2f} ± {results['eg']['cum_regret_std'][-1]:.2f}")
    print(f"  LinUCB:            {results['linucb']['cum_regret_mean'][-1]:.2f} ± {results['linucb']['cum_regret_std'][-1]:.2f}")
    print(f"  Thompson Sampling: {results['ts']['cum_regret_mean'][-1]:.2f} ± {results['ts']['cum_regret_std'][-1]:.2f}")

    print("\nCumulative Reward:")
    print(f"  ε-Greedy:          {results['eg']['cum_reward_mean'][-1]:.2f}")
    print(f"  LinUCB:            {results['linucb']['cum_reward_mean'][-1]:.2f}")
    print(f"  Thompson Sampling: {results['ts']['cum_reward_mean'][-1]:.2f}")

    print("\n% Optimal Action:")
    print(f"  ε-Greedy:          {results['eg']['optimal_pct'][-1]:.2f}%")
    print(f"  LinUCB:            {results['linucb']['optimal_pct'][-1]:.2f}%")
    print(f"  Thompson Sampling: {results['ts']['optimal_pct'][-1]:.2f}%")

    # Plot results
    plot_contextual_results(results, n_arms, context_dim)
