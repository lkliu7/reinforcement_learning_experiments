import random
import numpy as np
import matplotlib.pyplot as plt

# Experiment configuration
CONFIG = {
    'n_bandits': 10,  # Number of bandit arms
    'time_steps': 10000,  # Steps per experimental run
    'runs': 2000,  # Number of independent runs to average over
    'epsilon': 0.1,  # Probability of random exploration (epsilon-greedy)
    'mean_drift': 0.01,  # Standard deviation of random walk for bandit means
    'progress_interval': 100,  # How often to print progress updates
    'constant_step_size': 0.1,  # Fixed step size for exponential averaging
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon = CONFIG['epsilon']
progress_interval = CONFIG['progress_interval']
mean_drift = CONFIG['mean_drift']
constant_step_size = CONFIG['constant_step_size']

# MARK: Main experiment loop
def get_step_size(step_size_param, n):
    """Returns step size for value updates.

    Args:
        step_size_param: Either a function (for sample averaging: 1/n) or constant
        n: Action count for this bandit arm
    """
    if callable(step_size_param):
        return step_size_param(n)
    else:
        return step_size_param
        
def run_bandit_experiment(n_bandits = n_bandits,
                          step_size = lambda n: 1/n,  # Default: sample averaging
                          time_steps = time_steps, runs = runs, epsilon = 0,
                          mean_drift = 0,  # 0 = stationary, >0 = nonstationary
                          initial_estimate = 0,  # Optimistic initialization if > 0
                          progress_interval = progress_interval):

    # Accumulate results across all runs
    total_rewards = np.zeros(time_steps)
    total_optimal_actions = np.zeros(time_steps)

    for run in range(runs):

        # Initialize each run
        bandit_means = np.zeros(n_bandits)  # True reward means (unknown to agent)
        estimated_means = np.zeros(n_bandits) + initial_estimate  # Agent's estimates
        action_counts = np.zeros(n_bandits)  # Count of times each arm was pulled
        rewards = []  # Track rewards for this run
        optimal_actions = []  # Track whether optimal action was chosen

        for t in range(time_steps):
            # Epsilon-greedy action selection
            greedy_action = estimated_means.argmax()
            if random.random() < epsilon:
                action = np.random.randint(n_bandits)  # Explore randomly
            else:
                action = greedy_action  # Exploit best estimate

            action_counts[action] += 1

            # Generate reward: true mean + unit normal noise
            reward = bandit_means[action] + np.random.randn()

            # Update value estimate using incremental formula:
            # Q_new = Q_old + step_size * (reward - Q_old)
            estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) * get_step_size(step_size, action_counts[action])

            rewards.append(reward)

            # Track if we chose the optimal action
            optimal_bandit = bandit_means.argmax()
            if action == optimal_bandit:
                optimal_actions.append(1)
            else:
                optimal_actions.append(0)

            # Nonstationary environment: bandit means undergo random walk
            bandit_means += np.random.randn(n_bandits) * mean_drift

        total_rewards += np.array(rewards)
        total_optimal_actions += np.array(optimal_actions)
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average results across all runs
    total_rewards /= runs
    total_optimal_actions /= runs
    return total_rewards, total_optimal_actions

# MARK: Sample average
# Uses 1/n step size - gives equal weight to all past rewards
sample_average_rewards, sample_average_optimal_frequency = run_bandit_experiment(epsilon=epsilon, mean_drift=mean_drift)

# MARK: Exponential recency-weighted average
# Uses constant step size - gives more weight to recent rewards (better for nonstationary)
exponential_average_rewards, exponential_optimal_frequency = run_bandit_experiment(step_size=constant_step_size, epsilon=epsilon, mean_drift=mean_drift)

# Plot average rewards over time
plt.plot(range(1, time_steps+1), sample_average_rewards, label='sample average')
plt.plot(range(1, time_steps+1), exponential_average_rewards, label='exponential average')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title(f'average performance of different averaging methods')
plt.legend()
plt.show()

# Plot fraction of optimal actions chosen (skip first step where all actions are equally optimal)
plt.plot(range(2, time_steps+1), sample_average_optimal_frequency[1:], label='sample average')
plt.plot(range(2, time_steps+1), exponential_optimal_frequency[1:], label='exponential average')
plt.xlabel('steps')
plt.ylabel('optimal action ratio')
plt.title(f'average fraction methods chose optimal action')
plt.legend()
plt.show()
