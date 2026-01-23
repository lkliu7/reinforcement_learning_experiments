from bandit import Bandit
from gradient_agent import GradientAgent
from experiment_utils import run_experiment, plot_performance

# Replicate Sutton and Barto, Figure 2.5: Gradient Bandit Algorithm

# Experiment configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
    'progress_interval': 100,
    'bandit_mean_baseline': 4,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
progress_interval = CONFIG['progress_interval']
bandit_mean_baseline = CONFIG['bandit_mean_baseline']

results = {}

# Test gradient bandit with different alpha values and baseline settings
for alpha in [0.1, 0.4]:
    for include_baseline in [True, False]:
        print(f"Running gradient bandit with alpha={alpha}, baseline={include_baseline}")

        # Create bandit environment
        bandit = Bandit(n_bandits=n_bandits,
                       mean_baseline=bandit_mean_baseline,
                       mean_std=1,
                       reward_std=1)

        # Create gradient bandit agent
        agent = GradientAgent(actions=n_bandits,
                             alpha=alpha,
                             include_baseline=include_baseline)

        # Run experiment
        rewards, optimal_actions = run_experiment(bandit, agent, time_steps, runs, progress_interval)

        # Store results
        label = f'alpha={alpha}, baseline={include_baseline}'
        results[label] = (rewards, optimal_actions)

# Plot results
plot_performance(results, title_prefix="Gradient Bandit")