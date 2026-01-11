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
runs = CONFIG['runs']

ensemble_average_rewards = []
ensemble_optimal_ratio = []

for run in range(runs):
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

    ensemble_average_rewards.append(average_reward)
    ensemble_optimal_ratio.append(optimal_ratio)
    if run % 100 == 99:
        print(f'Completed run {run + 1}.')

ensemble_average_rewards = np.array(ensemble_average_rewards)
ensemble_average_rewards = np.sum(ensemble_average_rewards, axis=0) / runs
ensemble_optimal_ratio = np.array(ensemble_optimal_ratio)
ensemble_optimal_ratio = np.sum(ensemble_optimal_ratio, axis=0) / runs
