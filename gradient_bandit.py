import random
import numpy as np
import matplotlib.pyplot as plt

# Experiment configuration
CONFIG = {
    'n_bandits': 10,  # Number of bandit arms
    'time_steps': 1000,  # Steps per experimental run
    'runs': 2000,  # Number of independent runs to average over
    'progress_interval': 100,  # How often to print progress updates
    'constant_step_size': 0.1,  # Step-size parameter
    'bandit_mean_baseline': 4,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
progress_interval = CONFIG['progress_interval']
constant_step_size = CONFIG['constant_step_size']
bandit_mean_baseline = CONFIG['bandit_mean_baseline']

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

ensemble_total_rewards = {}
ensemble_optimal_actions = {}

for alpha in [0.1, 0.4]:
    for include_baseline in [True, False]:

        total_rewards = np.zeros(time_steps)
        total_optimal_actions = np.zeros(time_steps)

        for run in range(runs):

            # Initialize each run
            bandit_means = np.random.randn(n_bandits) + bandit_mean_baseline  # True reward means (unknown to agent)
            optimal_bandit = bandit_means.argmax()
            estimated_mean = 0 # Agent's estimate
            action_counts = np.zeros(n_bandits)  # Count of times each arm was pulled
            H = np.zeros(n_bandits) # Preference vector
            rewards = []  # Track rewards for this run
            optimal_actions = []  # Track whether optimal action was chosen

            for t in range(time_steps):
                H = H - np.max(H)
                dist = np.exp(H)
                dist = dist / np.sum(dist)

                action = np.random.choice(range(n_bandits), p=dist)
                action_vec = np.eye(n_bandits)[action]

                action_counts[action] += 1

                # Generate reward: true mean + unit normal noise
                reward = bandit_means[action] + np.random.randn()
                rewards.append(reward)
                
                # Update value estimate using incremental formula
                estimated_mean = estimated_mean + (reward - estimated_mean) / (t + 1)

                H += alpha * (reward - include_baseline * estimated_mean) * (action_vec - dist)

                # Track if we chose the optimal action
                if action == optimal_bandit:
                    optimal_actions.append(1)
                else:
                    optimal_actions.append(0)

            total_rewards += np.array(rewards)
            total_optimal_actions += np.array(optimal_actions)
            if run % progress_interval == progress_interval - 1:
                print(f'Completed run {run + 1}.')

        # Average results across all runs
        total_rewards /= runs
        total_optimal_actions /= runs

        ensemble_total_rewards[(alpha, include_baseline)] = total_rewards
        ensemble_optimal_actions[(alpha, include_baseline)] = total_optimal_actions

for alpha in [0.1, 0.4]:
    for include_baseline in [True, False]:
        plt.plot(range(1, time_steps+1), ensemble_total_rewards[(alpha, include_baseline)], label=f'gradient bandit with alpha = {alpha}, with baseline = {include_baseline}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title(f'')
plt.legend()
plt.show()

for alpha in [0.1, 0.4]:
    for include_baseline in [True, False]:
        plt.plot(range(1, time_steps+1), ensemble_optimal_actions[(alpha, include_baseline)], label=f'gradient bandit with alpha = {alpha}, with baseline = {include_baseline}')
plt.xlabel('steps')
plt.ylabel('optimal action ratio')
plt.title(f'')
plt.legend()
plt.show()
