import random
import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    'n_bandits': 10,
    'time_steps': 10000,
    'runs': 2000,
    'epsilon': 0.1,
    'mean_drift': 0.01,
    'progress_interval': 100,
    'step_size': 0.1,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon = CONFIG['epsilon']
progress_interval = CONFIG['progress_interval']
mean_drift = CONFIG['mean_drift']
step_size = CONFIG['step_size']

# MARK: Sample average

sample_average_rewards = np.zeros(time_steps)
sample_average_optimal_frequency = np.zeros(time_steps)

for run in range(runs):

    bandit_means = np.zeros(n_bandits)
    estimated_means = np.zeros(n_bandits)
    action_counts = np.zeros(n_bandits)
    rewards = []
    optimal_actions = []

    for t in range(time_steps):
        greedy_action = estimated_means.argmax()
        if random.random() < epsilon:
            action = np.random.randint(n_bandits)
        else:
            action = greedy_action
        action_counts[action] += 1
        reward = bandit_means[action] + np.random.randn()
        estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) / action_counts[action]
        rewards.append(reward)
        optimal_bandit = bandit_means.argmax()
        if action == optimal_bandit:
            optimal_actions.append(1)
        else:
            optimal_actions.append(0)
        bandit_means += np.random.randn(n_bandits) * mean_drift

    sample_average_rewards += np.array(rewards)
    sample_average_optimal_frequency += np.array(optimal_actions)
    if run % progress_interval == progress_interval - 1:
        print(f'Completed run {run + 1}.')

sample_average_rewards /= runs
sample_average_optimal_frequency /= runs

# MARK: Exponential recency-weighted average

exponential_average_rewards = np.zeros(time_steps)
exponential_optimal_frequency = np.zeros(time_steps)

for run in range(runs):

    bandit_means = np.zeros(n_bandits)
    estimated_means = np.zeros(n_bandits)
    rewards = []
    optimal_actions = []

    for t in range(time_steps):
        greedy_action = estimated_means.argmax()
        if random.random() < epsilon:
            action = np.random.randint(n_bandits)
        else:
            action = greedy_action
        reward = bandit_means[action] + np.random.randn()
        estimated_means[action] = estimated_means[action] + step_size * (reward - estimated_means[action])
        rewards.append(reward)
        optimal_bandit = bandit_means.argmax()
        if action == optimal_bandit:
            optimal_actions.append(1)
        else:
            optimal_actions.append(0)
        bandit_means += np.random.randn(n_bandits) * mean_drift

    exponential_average_rewards += np.array(rewards)
    exponential_optimal_frequency += np.array(optimal_actions)
    if run % progress_interval == progress_interval - 1:
        print(f'Completed run {run + 1}.')

exponential_average_rewards /= runs
exponential_optimal_frequency /= runs

plt.plot(range(1, time_steps+1), sample_average_rewards, label='sample average')
plt.plot(range(1, time_steps+1), exponential_average_rewards, label='exponential average')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title(f'average performance of different averaging methods')
plt.legend()
plt.show()

plt.plot(range(2, time_steps+1), sample_average_optimal_frequency[1:], label='sample average')
plt.plot(range(2, time_steps+1), exponential_optimal_frequency[1:], label='exponential average')
plt.xlabel('steps')
plt.ylabel('optimal action ratio')
plt.title(f'average fraction methods chose optimal action')
plt.legend()
plt.show()
