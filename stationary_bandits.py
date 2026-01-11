import random
import numpy as np

# MARK: Configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
    'epsilon': 0.01,
}

n_bandits = CONFIG['n_bandits']
steps = CONFIG['time_steps']

bandit_means = np.random.randn(n_bandits)
means = np.zeros(n_bandits)
rewards = np.zeros(n_bandits)
actions = np.zeros(n_bandits)
optimal_bandit = bandit_means.argmax()
average_reward = []
optimal_ratio = []

for t in range(steps):
    greedy_choice = means.argmax()
    choice = greedy_choice
    actions[choice] += 1
    reward = bandit_means[choice] + np.random.randn()
    rewards[choice] += reward
    means[choice] = rewards[choice] / actions[choice]
    average_reward.append(rewards.sum() / (t + 1))
    optimal_ratio.append(actions[optimal_bandit] / (t + 1))
