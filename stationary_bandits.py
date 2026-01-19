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
    'bias': 0,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon_list = CONFIG['epsilon_list']
progress_interval = CONFIG['progress_interval']
bias = CONFIG['bias']

cumulative_average_rewards = {}
cumulative_optimal_frequency = {}
for epsilon in epsilon_list:
    ensemble_average_rewards = np.zeros(time_steps)
    ensemble_optimal_frequency = np.zeros(time_steps)

    for run in range(runs):
        # Each bandit has a reward drawn from N(q*(a), 1) where q*(a) ~ N(0, 1)
        bandit_means = np.random.randn(n_bandits)
        estimated_means = np.zeros(n_bandits) + bias
        action_counts = np.zeros(n_bandits)
        optimal_bandit = bandit_means.argmax()
        rewards = []
        optimal_actions = []

        for t in range(time_steps):
            greedy_choice = estimated_means.argmax()
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = np.random.randint(n_bandits)  # Explore
            else:
                action = greedy_choice  # Exploit
            action_counts[action] += 1
            reward = bandit_means[action] + np.random.randn()  # Reward ~ N(q*(a), 1)
            estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) / action_counts[action]  # Sample average
            rewards.append(reward)  # Record of all rewards
            if action == optimal_bandit:
                optimal_actions.append(1)
            else:
                optimal_actions.append(0)

        ensemble_average_rewards += np.array(rewards)
        ensemble_optimal_frequency += np.array(optimal_actions)
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average performance across all runs for this epsilon value
    cumulative_average_rewards[epsilon] = ensemble_average_rewards / runs
    cumulative_optimal_frequency[epsilon] = ensemble_optimal_frequency / runs

for epsilon in epsilon_list:
    plt.plot(range(1, time_steps+1), cumulative_average_rewards[epsilon], label=f'epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title(f'average performance of epsilon-greedy methods')
plt.legend()
plt.show()

for epsilon in epsilon_list:
    plt.plot(range(1, time_steps+1), cumulative_optimal_frequency[epsilon], label=f'epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('optimal action ratio')
plt.title(f'average fraction epsilon-greedy methods chose optimal action')
plt.legend()
plt.show()