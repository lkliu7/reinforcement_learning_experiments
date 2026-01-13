import random
import numpy as np
import matplotlib.pyplot as plt

# 10-armed testbed experiment: compares epsilon-greedy methods with different exploration rates

# MARK: Configuration
CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
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
        # Each bandit has a reward drawn from N(q*(a), 1) where q*(a) ~ N(0, 1)
        bandit_means = np.random.randn(n_bandits)
        estimated_means = np.zeros(n_bandits)
        total_rewards = np.zeros(n_bandits)
        action_counts = np.zeros(n_bandits)
        optimal_bandit = bandit_means.argmax()
        average_reward = []
        optimal_ratio = []
        running_total = 0

        for t in range(time_steps):
            greedy_choice = estimated_means.argmax()
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = np.random.randint(n_bandits)  # Explore
            else:
                action = greedy_choice  # Exploit
            action_counts[action] += 1
            reward = bandit_means[action] + np.random.randn()  # Reward ~ N(q*(a), 1)
            total_rewards[action] += reward
            running_total += reward
            estimated_means[action] = total_rewards[action] / action_counts[action]  # Sample average
            average_reward.append(running_total / (t + 1))  # Cumulative average reward
            optimal_ratio.append(action_counts[optimal_bandit] / (t + 1))  # Fraction of optimal actions

        ensemble_average_rewards.append(average_reward)
        ensemble_optimal_ratio.append(optimal_ratio)
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average performance across all runs for this epsilon value
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