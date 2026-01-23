from bandit import Bandit
from greedy_agent import GreedyAgent, sample_average_step_size
from experiment_utils import run_experiment, plot_performance

# Experiment configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 10000,
    'runs': 2000,
    'epsilon': 0.1,
    'mean_drift': 0.01,  # Standard deviation of random walk for bandit means
    'progress_interval': 100,
    'constant_step_size': 0.1,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon = CONFIG['epsilon']
mean_drift = CONFIG['mean_drift']
progress_interval = CONFIG['progress_interval']
constant_step_size = CONFIG['constant_step_size']

results = {}

# Sample average method (gives equal weight to all past rewards)
print("Running sample average method")
bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=0,
                reward_std=1, mean_drift_std=mean_drift)
sample_agent = GreedyAgent(actions=n_bandits, epsilon=epsilon,
                          step_size=sample_average_step_size)
sample_rewards, sample_optimal = run_experiment(bandit, sample_agent,
                                               time_steps, runs, progress_interval)
results['sample average'] = (sample_rewards, sample_optimal)

# Exponential recency-weighted average (gives more weight to recent rewards)
print("Running exponential average method")
bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=0,
                reward_std=1, mean_drift_std=mean_drift)
exponential_agent = GreedyAgent(actions=n_bandits, epsilon=epsilon,
                               step_size=constant_step_size)
exp_rewards, exp_optimal = run_experiment(bandit, exponential_agent,
                                         time_steps, runs, progress_interval)
results['exponential average'] = (exp_rewards, exp_optimal)

# Plot comparison
plot_performance(results,
                title_prefix="Nonstationary Bandits",
                ylabel_rewards="average reward",
                ylabel_optimal="optimal action ratio")