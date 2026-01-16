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

sample_average_rewards = []
sample_average_optimal_ratio = []

for run in range(runs):

    bandit_means = np.zeros(n_bandits)
    estimated_means = np.zeros(n_bandits)
    total_rewards = np.zeros(n_bandits)
    action_counts = np.zeros(n_bandits)
    average_reward = []
    optimal_ratio = []
    optimal_action_count = 0
    running_total = 0

    for t in range(time_steps):
        greedy_action = estimated_means.argmax()
        if random.random() < epsilon:
            action = np.random.randint(n_bandits)
        else:
            action = greedy_action
        action_counts[action] += 1
        reward = bandit_means[action] + np.random.randn()
        total_rewards[action] += reward
        running_total += reward
        estimated_means[action] = total_rewards[action] / action_counts[action]
        average_reward.append(running_total / (t + 1))
        optimal_bandit = bandit_means.argmax()
        if action == optimal_bandit:
            optimal_action_count += 1
        optimal_ratio.append(optimal_action_count / (t + 1))
        bandit_means += np.random.randn(n_bandits) * mean_drift

    sample_average_rewards.append(average_reward)
    sample_average_optimal_ratio.append(optimal_ratio)
    if run % progress_interval == progress_interval - 1:
        print(f'Completed run {run + 1}.')

sample_average_rewards = np.array(sample_average_rewards).mean(axis=0)
sample_average_optimal_ratio = np.array(sample_average_optimal_ratio).mean(axis=0)

# MARK: Exponential recency-weighted average

exponential_average_rewards = []
exponential_optimal_ratio = []

for run in range(runs):

    bandit_means = np.zeros(n_bandits)
    estimated_means = np.zeros(n_bandits)
    action_counts = np.zeros(n_bandits)
    average_reward = []
    optimal_ratio = []
    optimal_action_count = 0
    running_total = 0

    for t in range(time_steps):
        greedy_action = estimated_means.argmax()
        if random.random() < epsilon:
            action = np.random.randint(n_bandits)
        else:
            action = greedy_action
        action_counts[action] += 1
        reward = bandit_means[action] + np.random.randn()
        running_total += reward
        estimated_means[action] = estimated_means[action] + step_size * (reward - estimated_means[action])
        average_reward.append(running_total / (t + 1))
        optimal_bandit = bandit_means.argmax()
        if action == optimal_bandit:
            optimal_action_count += 1
        optimal_ratio.append(optimal_action_count / (t + 1))
        bandit_means += np.random.randn(n_bandits) * mean_drift

    exponential_average_rewards.append(average_reward)
    exponential_optimal_ratio.append(optimal_ratio)
    if run % progress_interval == progress_interval - 1:
        print(f'Completed run {run + 1}.')

exponential_average_rewards = np.array(exponential_average_rewards).mean(axis=0)
exponential_optimal_ratio = np.array(exponential_optimal_ratio).mean(axis=0)

plt.plot(range(1, time_steps+1), sample_average_rewards)
plt.plot(range(1, time_steps+1), exponential_average_rewards)
plt.show()

plt.plot(range(2, time_steps+1), sample_average_optimal_ratio[1:])
plt.plot(range(2, time_steps+1), exponential_optimal_ratio[1:])
plt.show()
