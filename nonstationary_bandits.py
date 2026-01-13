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
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon = CONFIG['epsilon']
progress_interval = CONFIG['progress_interval']
mean_drift = CONFIG['mean_drift']

bandit_means = np.zeros(n_bandits)
estimated_means = np.zeros(n_bandits)
total_rewards = np.zeros(n_bandits)
action_counts = np.zeros(n_bandits)
average_reward = []
optimal_ratio = []
running_total = 0

for t in range(time_steps):
    greedy_choice = estimated_means.argmax()
    if random.random() < epsilon:
        action = np.random.randint(n_bandits)
    else:
        action = greedy_choice
    action_counts[action] += 1
    reward = bandit_means[action] + np.random.randn()
    total_rewards[action] += reward
    running_total += reward
    estimated_means[action] = total_rewards[action] / action_counts[action]
    average_reward.append(running_total / (t + 1))
    optimal_bandit = bandit_means.argmax()
    optimal_ratio.append(action_counts[optimal_bandit] / (t + 1))
    bandit_means += np.random.randn(n_bandits) * mean_drift