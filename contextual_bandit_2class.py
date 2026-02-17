import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TwoClassContextualBandit:
    """
    Contextual bandit with 2 customer classes
    Each class has different reward probabilities for each arm
    """
    def __init__(self, n_arms=3):
        self.n_arms = n_arms
        self.n_classes = 2

        # Reward probabilities for each (class, arm) pair
        # Class 0: prefers arm 0
        # Class 1: prefers arm 2
        self.reward_probs = np.array([
            [0.8, 0.5, 0.3],  # Class 0 (e.g., Young customers)
            [0.3, 0.5, 0.8]   # Class 1 (e.g., Old customers)
        ])

        print(f"\nTrue reward probabilities:")
        print(f"  Class 0 (Young): {self.reward_probs[0]}")
        print(f"  Class 1 (Old):   {self.reward_probs[1]}")
        print(f"  Optimal arms: Class 0 → Arm {np.argmax(self.reward_probs[0])}, "
              f"Class 1 → Arm {np.argmax(self.reward_probs[1])}\n")

    def get_context(self):
        """Randomly sample a customer class (0 or 1)"""
        return np.random.randint(self.n_classes)

    def get_reward(self, context, arm):
        """Get Bernoulli reward based on context and arm"""
        prob = self.reward_probs[context, arm]
        return np.random.binomial(1, prob)

    def get_optimal_arm(self, context):
        """Return optimal arm for given context"""
        return np.argmax(self.reward_probs[context])

    def get_optimal_prob(self, context):
        """Return optimal reward probability for given context"""
        return np.max(self.reward_probs[context])


class EpsilonGreedyContextual:
    """
    ε-greedy that learns separate Q-values for each (context, arm) pair
    """
    def __init__(self, n_arms, n_classes, epsilon=0.1):
        self.n_arms = n_arms
        self.n_classes = n_classes
        self.epsilon = epsilon

        # Q-values: Q[context][arm]
        self.Q = np.zeros((n_classes, n_arms))
        self.counts = np.zeros((n_classes, n_arms))

    def select_arm(self, context):
        """ε-greedy selection given context"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.Q[context])

    def update(self, context, arm, reward):
        """Update Q-value for (context, arm)"""
        self.counts[context, arm] += 1
        n = self.counts[context, arm]
        # Incremental average
        self.Q[context, arm] += (reward - self.Q[context, arm]) / n


class UCBContextual:
    """
    UCB with separate values for each (context, arm) pair
    """
    def __init__(self, n_arms, n_classes, c=2.0):
        self.n_arms = n_arms
        self.n_classes = n_classes
        self.c = c

        self.Q = np.zeros((n_classes, n_arms))
        self.counts = np.zeros((n_classes, n_arms))
        self.t = np.zeros(n_classes)  # Time step per class

    def select_arm(self, context):
        """UCB selection given context"""
        self.t[context] += 1

        # Pull each arm at least once for this context
        for arm in range(self.n_arms):
            if self.counts[context, arm] == 0:
                return arm

        # UCB formula
        ucb_values = self.Q[context] + self.c * np.sqrt(
            np.log(self.t[context]) / self.counts[context]
        )
        return np.argmax(ucb_values)

    def update(self, context, arm, reward):
        """Update Q-value for (context, arm)"""
        self.counts[context, arm] += 1
        n = self.counts[context, arm]
        self.Q[context, arm] += (reward - self.Q[context, arm]) / n


class ThompsonSamplingContextual:
    """
    Thompson Sampling with Beta priors for each (context, arm) pair
    """
    def __init__(self, n_arms, n_classes):
        self.n_arms = n_arms
        self.n_classes = n_classes

        # Beta(alpha, beta) for each (context, arm)
        self.alpha = np.ones((n_classes, n_arms))
        self.beta = np.ones((n_classes, n_arms))

    def select_arm(self, context):
        """Sample from Beta distributions and choose best"""
        samples = np.random.beta(self.alpha[context], self.beta[context])
        return np.argmax(samples)

    def update(self, context, arm, reward):
        """Update Beta distribution for (context, arm)"""
        self.alpha[context, arm] += reward
        self.beta[context, arm] += (1 - reward)


class ContextBlindBaseline:
    """
    Baseline that ignores context (standard UCB)
    Shows importance of using context
    """
    def __init__(self, n_arms, n_classes):
        self.n_arms = n_arms

        self.Q = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.t = 0

    def select_arm(self, context):
        """UCB ignoring context"""
        self.t += 1

        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.Q + 2.0 * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)

    def update(self, context, arm, reward):
        """Update ignoring context"""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.Q[arm] += (reward - self.Q[arm]) / n


def run_experiment(algorithm, bandit, n_steps):
    """Run contextual bandit experiment"""
    cumulative_regrets = np.zeros(n_steps)
    instantaneous_regrets = np.zeros(n_steps)
    cumulative_rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)

    cumulative_regret = 0
    cumulative_reward = 0

    for t in range(n_steps):
        # Get context (customer class)
        context = bandit.get_context()

        # Select arm
        arm = algorithm.select_arm(context)

        # Get reward
        reward = bandit.get_reward(context, arm)

        # Update
        algorithm.update(context, arm, reward)

        # Calculate regret
        optimal_prob = bandit.get_optimal_prob(context)
        actual_prob = bandit.reward_probs[context, arm]
        inst_regret = optimal_prob - actual_prob

        optimal_arm = bandit.get_optimal_arm(context)

        cumulative_regret += inst_regret
        cumulative_reward += reward

        instantaneous_regrets[t] = inst_regret
        cumulative_regrets[t] = cumulative_regret
        cumulative_rewards[t] = cumulative_reward
        optimal_actions[t] = 1 if arm == optimal_arm else 0

    return cumulative_regrets, instantaneous_regrets, cumulative_rewards, optimal_actions


def compare_algorithms(n_arms=3, n_steps=5000, n_runs=100):
    """Compare contextual bandit algorithms"""

    # Initialize storage
    results = {}
    for alg_name in ['eg', 'ucb', 'ts', 'blind']:
        results[alg_name] = {
            'cum_regrets': np.zeros((n_runs, n_steps)),
            'inst_regrets': np.zeros((n_runs, n_steps)),
            'cum_rewards': np.zeros((n_runs, n_steps)),
            'optimal_actions': np.zeros((n_runs, n_steps))
        }

    for run in range(n_runs):
        if (run + 1) % 20 == 0:
            print(f"  Run {run + 1}/{n_runs}")

        # Create bandit
        bandit = TwoClassContextualBandit(n_arms)

        # ε-greedy
        eg = EpsilonGreedyContextual(n_arms, n_classes=2, epsilon=0.1)
        results['eg']['cum_regrets'][run], results['eg']['inst_regrets'][run], \
        results['eg']['cum_rewards'][run], results['eg']['optimal_actions'][run] = \
            run_experiment(eg, bandit, n_steps)

        # UCB
        ucb = UCBContextual(n_arms, n_classes=2, c=2.0)
        results['ucb']['cum_regrets'][run], results['ucb']['inst_regrets'][run], \
        results['ucb']['cum_rewards'][run], results['ucb']['optimal_actions'][run] = \
            run_experiment(ucb, bandit, n_steps)

        # Thompson Sampling
        ts = ThompsonSamplingContextual(n_arms, n_classes=2)
        results['ts']['cum_regrets'][run], results['ts']['inst_regrets'][run], \
        results['ts']['cum_rewards'][run], results['ts']['optimal_actions'][run] = \
            run_experiment(ts, bandit, n_steps)

        # Context-blind baseline
        blind = ContextBlindBaseline(n_arms, n_classes=2)
        results['blind']['cum_regrets'][run], results['blind']['inst_regrets'][run], \
        results['blind']['cum_rewards'][run], results['blind']['optimal_actions'][run] = \
            run_experiment(blind, bandit, n_steps)

    # Calculate statistics
    final_results = {}
    for alg_name in ['eg', 'ucb', 'ts', 'blind']:
        final_results[alg_name] = {
            'cum_regret_mean': np.mean(results[alg_name]['cum_regrets'], axis=0),
            'cum_regret_std': np.std(results[alg_name]['cum_regrets'], axis=0),
            'inst_regret_mean': np.mean(results[alg_name]['inst_regrets'], axis=0),
            'cum_reward_mean': np.mean(results[alg_name]['cum_rewards'], axis=0),
            'optimal_pct': np.mean(results[alg_name]['optimal_actions'], axis=0) * 100
        }

    return final_results


def plot_results(results):
    """Plot comparison results"""
    n_steps = len(results['eg']['cum_regret_mean'])
    steps = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Contextual Bandit: 2 Customer Classes, 3 Arms\n'
                 'Class 0 prefers Arm 0 (0.8), Class 1 prefers Arm 2 (0.8)',
                 fontsize=14, fontweight='bold')

    colors = {'eg': 'C0', 'ucb': 'C1', 'ts': 'C2', 'blind': 'C3'}
    labels = {
        'eg': 'ε-Greedy (context-aware)',
        'ucb': 'UCB (context-aware)',
        'ts': 'Thompson Sampling (context-aware)',
        'blind': 'UCB (context-blind)'
    }

    # Cumulative Regret
    ax = axes[0, 0]
    for alg in ['eg', 'ucb', 'ts', 'blind']:
        ax.plot(steps, results[alg]['cum_regret_mean'],
                label=labels[alg], linewidth=2, color=colors[alg])
        ax.fill_between(steps,
                         results[alg]['cum_regret_mean'] - results[alg]['cum_regret_std'],
                         results[alg]['cum_regret_mean'] + results[alg]['cum_regret_std'],
                         alpha=0.2, color=colors[alg])

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Cumulative Regret', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cumulative Reward
    ax = axes[0, 1]
    for alg in ['eg', 'ucb', 'ts', 'blind']:
        ax.plot(steps, results[alg]['cum_reward_mean'],
                label=labels[alg], linewidth=2, color=colors[alg])

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Instantaneous Regret (smoothed)
    ax = axes[1, 0]
    window = 50
    for alg in ['eg', 'ucb', 'ts', 'blind']:
        smoothed = np.convolve(results[alg]['inst_regret_mean'],
                               np.ones(window)/window, mode='valid')
        ax.plot(steps[:len(smoothed)], smoothed,
                label=labels[alg], linewidth=2, color=colors[alg])

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Instantaneous Regret')
    ax.set_title(f'Instantaneous Regret (smoothed, window={window})', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # % Optimal Action
    ax = axes[1, 1]
    for alg in ['eg', 'ucb', 'ts', 'blind']:
        ax.plot(steps, results[alg]['optimal_pct'],
                label=labels[alg], linewidth=2, color=colors[alg])

    ax.set_xlabel('Time Steps')
    ax.set_ylabel('% Optimal Action')
    ax.set_title('Percentage of Optimal Actions', fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('contextual_bandit_2class.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'contextual_bandit_2class.png'")


if __name__ == "__main__":
    n_arms = 3
    n_steps = 5000
    n_runs = 100

    print("=" * 60)
    print("Contextual Bandit: 2 Customer Classes")
    print("=" * 60)
    print(f"Number of arms: {n_arms}")
    print(f"Number of customer classes: 2")
    print(f"Time steps: {n_steps}")
    print(f"Number of runs: {n_runs}")

    # Run comparison
    print("\nRunning experiments...")
    results = compare_algorithms(n_arms, n_steps, n_runs)

    # Print final results
    print(f"\nFinal Results (after {n_steps} steps):")
    print("-" * 60)
    print("Cumulative Regret:")
    print(f"  ε-Greedy (context-aware):  {results['eg']['cum_regret_mean'][-1]:.2f} ± {results['eg']['cum_regret_std'][-1]:.2f}")
    print(f"  UCB (context-aware):       {results['ucb']['cum_regret_mean'][-1]:.2f} ± {results['ucb']['cum_regret_std'][-1]:.2f}")
    print(f"  Thompson (context-aware):  {results['ts']['cum_regret_mean'][-1]:.2f} ± {results['ts']['cum_regret_std'][-1]:.2f}")
    print(f"  UCB (context-BLIND):       {results['blind']['cum_regret_mean'][-1]:.2f} ± {results['blind']['cum_regret_std'][-1]:.2f}")

    print("\nCumulative Reward:")
    print(f"  ε-Greedy (context-aware):  {results['eg']['cum_reward_mean'][-1]:.2f}")
    print(f"  UCB (context-aware):       {results['ucb']['cum_reward_mean'][-1]:.2f}")
    print(f"  Thompson (context-aware):  {results['ts']['cum_reward_mean'][-1]:.2f}")
    print(f"  UCB (context-BLIND):       {results['blind']['cum_reward_mean'][-1]:.2f}")

    print("\n% Optimal Action:")
    print(f"  ε-Greedy (context-aware):  {results['eg']['optimal_pct'][-1]:.2f}%")
    print(f"  UCB (context-aware):       {results['ucb']['optimal_pct'][-1]:.2f}%")
    print(f"  Thompson (context-aware):  {results['ts']['optimal_pct'][-1]:.2f}%")
    print(f"  UCB (context-BLIND):       {results['blind']['optimal_pct'][-1]:.2f}%")

    plot_results(results)
