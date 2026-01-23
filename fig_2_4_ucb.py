from bandit import Bandit
from greedy_agent import GreedyAgent
from ucb_agent import UCBAgent
from experiment_utils import run_experiment, plot_performance

# Experiment configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
    'epsilon': 0.1,
    'progress_interval': 100,
    'constant_step_size': 0.1,
    'c': 2,  # UCB confidence parameter
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon = CONFIG['epsilon']
progress_interval = CONFIG['progress_interval']
c = CONFIG['c']

results = {}

# UCB Agent
print("Running UCB agent")
bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=1, reward_std=1)
ucb_agent = UCBAgent(actions=n_bandits, c=c)
ucb_rewards, ucb_optimal = run_experiment(bandit, ucb_agent, time_steps, runs, progress_interval)
results['UCB'] = (ucb_rewards, ucb_optimal)

# Epsilon-greedy Agent
print("Running epsilon-greedy agent")
bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=1, reward_std=1)
greedy_agent = GreedyAgent(actions=n_bandits, epsilon=epsilon)
greedy_rewards, greedy_optimal = run_experiment(bandit, greedy_agent, time_steps, runs, progress_interval)
results[f'epsilon={epsilon}'] = (greedy_rewards, greedy_optimal)

# Plot comparison
plot_performance(results, title_prefix="UCB vs Epsilon-Greedy")