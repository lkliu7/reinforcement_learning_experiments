import random
import numpy as np
import matplotlib.pyplot as plt

# MARK: Configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
    'epsilon': 0.01,
    'epsilon_list': [0, 0.01, 0.1],
    'progress_interval': 100,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon_list = CONFIG['epsilon_list']
progress_interval = CONFIG['progress_interval']

cumulative_average_rewards = {}
cumulative_optimal_ratio = {}
for epsilon in epsilon_list:
    ensemble_average_rewards = []
    ensemble_optimal_ratio = []

    for run in range(runs):
        bandit_means = np.random.randn(n_bandits)
        estimated_means = np.zeros(n_bandits)
        total_rewards = np.zeros(n_bandits)
        action_counts = np.zeros(n_bandits)
        optimal_bandit = bandit_means.argmax()
        average_reward = []
        optimal_ratio = []

        for t in range(time_steps):
            greedy_choice = estimated_means.argmax()
            if random.random() < epsilon:
                action = np.random.randint(n_bandits)
            else:
                action = greedy_choice
            action_counts[action] += 1
            reward = bandit_means[action] + np.random.randn()
            total_rewards[action] += reward
            estimated_means[action] = total_rewards[action] / action_counts[action]
            average_reward.append(total_rewards.sum() / (t + 1))
            optimal_ratio.append(action_counts[optimal_bandit] / (t + 1))

        ensemble_average_rewards.append(average_reward)
        ensemble_optimal_ratio.append(optimal_ratio)
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    ensemble_average_rewards = np.array(ensemble_average_rewards)
    cumulative_average_rewards[epsilon] = np.sum(ensemble_average_rewards, axis=0) / runs
    ensemble_optimal_ratio = np.array(ensemble_optimal_ratio)
    cumulative_optimal_ratio[epsilon] = np.sum(ensemble_optimal_ratio, axis=0) / runs

for epsilon in epsilon_list:
    plt.plot(range(1, time_steps+1), cumulative_average_rewards[epsilon], label=f'epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title(f'average performance of epsilon-greedy methods')
plt.legend()
plt.show()

for epsilon in epsilon_list:
    plt.plot(range(1, time_steps+1), cumulative_optimal_ratio[epsilon], label=f'epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('optimal action ratio')
plt.title(f'average fraction epsilon-greedy methods chose optimal action')
plt.legend()
plt.show()