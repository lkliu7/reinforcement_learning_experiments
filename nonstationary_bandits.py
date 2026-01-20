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
    'constant_step_size': 0.1,
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
    if callable(step_size_param):
        return step_size_param(n)
    else:
        return step_size_param
        
def run_bandit_experiment(n_bandits = n_bandits,
                          step_size = lambda n: 1/n,
                          time_steps = time_steps, runs = runs, epsilon = 0,
                          mean_drift = 0,
                          initial_estimate = 0,
                          progress_interval = progress_interval):

    total_rewards = np.zeros(time_steps)
    total_optimal_actions = np.zeros(time_steps)
    for run in range(runs):

        bandit_means = np.zeros(n_bandits)
        estimated_means = np.zeros(n_bandits) + initial_estimate
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
            estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) * get_step_size(step_size, action_counts[action])
            rewards.append(reward)
            optimal_bandit = bandit_means.argmax()
            if action == optimal_bandit:
                optimal_actions.append(1)
            else:
                optimal_actions.append(0)
            bandit_means += np.random.randn(n_bandits) * mean_drift

        total_rewards += np.array(rewards)
        total_optimal_actions += np.array(optimal_actions)
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    average_rewards = total_rewards / runs
    optimal_frequency = total_optimal_actions / runs
    return average_rewards, optimal_frequency

# MARK: Sample average

sample_average_rewards, sample_average_optimal_frequency = run_bandit_experiment(epsilon=epsilon, mean_drift=mean_drift)

# MARK: Exponential recency-weighted average

exponential_average_rewards, exponential_optimal_frequency = run_bandit_experiment(step_size=constant_step_size, epsilon=epsilon, mean_drift=mean_drift)

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
