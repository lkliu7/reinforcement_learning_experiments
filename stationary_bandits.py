from bandit import Bandit
from greedy_agent import GreedyAgent
from experiment_utils import run_experiment, plot_performance

# Experiment configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
    'epsilon_list': [0, 0.01, 0.1],
    'progress_interval': 100,
    'bias': 0,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon_list = CONFIG['epsilon_list']
progress_interval = CONFIG['progress_interval']
bias = CONFIG['bias']

results = {}

# Test epsilon-greedy methods with different exploration rates
for epsilon in epsilon_list:
    print(f"Running epsilon-greedy with epsilon={epsilon}")

    # Create stationary bandit environment
    bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=1, reward_std=1)

    # Create epsilon-greedy agent
    agent = GreedyAgent(actions=n_bandits, epsilon=epsilon, initial_estimate=bias)

    # Run experiment
    rewards, optimal_actions = run_experiment(bandit, agent, time_steps, runs, progress_interval)

    # Store results
    results[f'epsilon={epsilon}'] = (rewards, optimal_actions)

# Plot results
plot_performance(results, title_prefix="Epsilon-Greedy Comparison")