import random
import numpy as np
import matplotlib.pyplot as plt

# Experiment configuration
CONFIG = {
    'n_bandits': 10,  # Number of bandit arms
    'time_steps': 1000,  # Steps per experimental run
    'runs': 2000,  # Number of independent runs to average over
    'epsilon': 0.1,  # Probability of random exploration (epsilon-greedy)
    'progress_interval': 100,  # How often to print progress updates
    'constant_step_size': 0.1,  # Fixed step size for exponential averaging
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon = CONFIG['epsilon']
progress_interval = CONFIG['progress_interval']
constant_step_size = CONFIG['constant_step_size']
initial_estimate = 0
c = 2

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

total_rewards = np.zeros(time_steps)
total_optimal_actions = np.zeros(time_steps)

for run in range(runs):

    # Initialize each run
    bandit_means = np.random.randn(n_bandits)  # True reward means (unknown to agent)
    estimated_means = np.zeros(n_bandits) + initial_estimate  # Agent's estimates
    action_counts = np.zeros(n_bandits)  # Count of times each arm was pulled
    rewards = []  # Track rewards for this run
    optimal_actions = []  # Track whether optimal action was chosen

    for t in range(time_steps):
        ucb_values = np.where(action_counts > 0, estimated_means + c * np.sqrt(np.log(t + 1) / action_counts), np.inf)
        action = np.argmax(ucb_values)

        action_counts[action] += 1

        # Generate reward: true mean + unit normal noise
        reward = bandit_means[action] + np.random.randn()

        # Update value estimate using incremental formula:
        # Q_new = Q_old + step_size * (reward - Q_old)
        estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) / action_counts[action]

        rewards.append(reward)

        # Track if we chose the optimal action
        optimal_bandit = bandit_means.argmax()
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

ensemble_average_rewards = np.zeros(time_steps)
ensemble_optimal_frequency = np.zeros(time_steps)

for run in range(runs):
    # Each bandit has a reward drawn from N(q*(a), 1) where q*(a) ~ N(0, 1)
    bandit_means = np.random.randn(n_bandits)
    estimated_means = np.zeros(n_bandits)
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

ensemble_average_rewards /= runs
ensemble_optimal_frequency /= runs

plt.plot(range(1, time_steps+1), total_rewards, label='UCB')
plt.plot(range(1, time_steps+1), ensemble_average_rewards, label=f'epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('average reward')
plt.title(f'')
plt.legend()
plt.show()

plt.plot(range(2, time_steps+1), total_optimal_actions[1:], label='UCB')
plt.plot(range(1, time_steps+1), ensemble_optimal_frequency, label=f'epsilon = {epsilon}')
plt.xlabel('steps')
plt.ylabel('optimal action ratio')
plt.title(f'')
plt.legend()
plt.show()

