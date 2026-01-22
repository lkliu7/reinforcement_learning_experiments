import random
import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit

# Experiment configuration
CONFIG = {
    'n_bandits': 10,  # Number of bandit arms
    'time_steps': 1000,  # Steps per experimental run
    'runs': 2000,  # Number of independent runs to average over
    'epsilon_values': [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],  # Probability of random exploration (epsilon-greedy)
    'gradient_bandit_step_sizes': [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    'ucb_c_values': [1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    'greedy_optimistic_initial_values': [1/4, 1/2, 1, 2, 4],
    'progress_interval': 500,  # How often to print progress updates
    'constant_step_size': 0.1,  # Fixed step size for exponential averaging
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon_values = CONFIG['epsilon_values']
greedy_optimistic_initial_values = CONFIG['greedy_optimistic_initial_values']
gradient_bandit_step_sizes = CONFIG['gradient_bandit_step_sizes']
ucb_c_values = CONFIG['ucb_c_values']
progress_interval = CONFIG['progress_interval']


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
        
def run_bandit_experiment(n_bandits = n_bandits,
                          step_size = lambda n: 1/n,  # Default: sample averaging
                          time_steps = time_steps, runs = runs, epsilon = 0,
                          mean_drift = 0,  # 0 = stationary, >0 = nonstationary
                          initial_estimate = 0,  # Optimistic initialization if > 0
                          progress_interval = progress_interval):

    # Accumulate results across all runs
    bandit = Bandit(n_bandits=n_bandits, mean_drift=mean_drift)
    average_rewards_across_runs = 0

    for run in range(runs):

        # Initialize each run
        estimated_means = np.zeros(n_bandits) + initial_estimate  # Agent's estimates
        action_counts = np.zeros(n_bandits)  # Count of times each arm was pulled
        total_rewards = 0  # Track rewards for this run

        for t in range(time_steps):
            # Epsilon-greedy action selection
            greedy_action = estimated_means.argmax()
            if random.random() < epsilon:
                action = np.random.randint(n_bandits)  # Explore randomly
            else:
                action = greedy_action  # Exploit best estimate

            action_counts[action] += 1

            # Generate reward: true mean + unit normal noise
            reward = bandit.pull(action)

            # Update value estimate using incremental formula:
            # Q_new = Q_old + step_size * (reward - Q_old)
            estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) * get_step_size(step_size, action_counts[action])

            total_rewards += reward
        
        bandit.reset()

        average_rewards_across_runs += total_rewards / time_steps
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average results across all runs
    average_rewards_across_runs /= runs
    return average_rewards_across_runs

def run_bandit_experiment_ucb(n_bandits = n_bandits,
                          step_size = lambda n: 1/n,  # Default: sample averaging
                          time_steps = time_steps, runs = runs, c = 1,
                          mean_drift = 0,  # 0 = stationary, >0 = nonstationary
                          initial_estimate = 0,  # Optimistic initialization if > 0
                          progress_interval = progress_interval):

    # Accumulate results across all runs
    bandit = Bandit(n_bandits=n_bandits, mean_drift=mean_drift)
    average_rewards_across_runs = 0

    for run in range(runs):

        # Initialize each run
        estimated_means = np.zeros(n_bandits) + initial_estimate  # Agent's estimates
        action_counts = np.zeros(n_bandits)  # Count of times each arm was pulled
        total_rewards = 0  # Track rewards for this run

        for t in range(time_steps):
            # Epsilon-greedy action selection
            ucb_values = np.where(action_counts > 0, estimated_means + c * np.sqrt(np.log(t + 1) / action_counts), np.inf)
            action = np.argmax(ucb_values)

            action_counts[action] += 1

            # Generate reward: true mean + unit normal noise
            reward = bandit.pull(action)

            # Update value estimate using incremental formula:
            # Q_new = Q_old + step_size * (reward - Q_old)
            estimated_means[action] = estimated_means[action] + (reward - estimated_means[action]) * get_step_size(step_size, action_counts[action])

            total_rewards += reward
        
        bandit.reset()

        average_rewards_across_runs += total_rewards / time_steps
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average results across all runs
    average_rewards_across_runs /= runs
    return average_rewards_across_runs

def run_bandit_experiment_grad(n_bandits = n_bandits,
                          step_size = 0.1,  # Default: sample averaging
                          time_steps = time_steps, runs = runs,
                          include_baseline = True,
                          mean_drift = 0,  # 0 = stationary, >0 = nonstationary
                          progress_interval = progress_interval):

    # Accumulate results across all runs
    bandit = Bandit(n_bandits=n_bandits, mean_drift=mean_drift)
    average_rewards_across_runs = 0

    for run in range(runs):

        # Initialize each run
        estimated_mean = 0  # Agent's estimates
        action_counts = np.zeros(n_bandits)  # Count of times each arm was pulled
        H = np.zeros(n_bandits) # Preference vector
        total_rewards = 0  # Track rewards for this run

        for t in range(time_steps):
            # Epsilon-greedy action selection
            H = H - np.max(H)
            dist = np.exp(H)
            dist = dist / np.sum(dist)

            action = np.random.choice(range(n_bandits), p=dist)
            action_vec = np.eye(n_bandits)[action]

            action_counts[action] += 1

            # Generate reward: true mean + unit normal noise
            reward = bandit.pull(action)

            # Update value estimate using incremental formula:
            # Q_new = Q_old + step_size * (reward - Q_old)
            estimated_mean = estimated_mean + (reward - estimated_mean) / (t + 1)

            H += step_size * (reward - include_baseline * estimated_mean) * (action_vec - dist)

            total_rewards += reward
        
        bandit.reset()

        average_rewards_across_runs += total_rewards / time_steps
        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average results across all runs
    average_rewards_across_runs /= runs
    return average_rewards_across_runs

epsilon_greedy_average_rewards = {}
greedy_optimistic_average_rewards = {}
ucb_average_rewards = {}
gradient_bandit_average_rewards = {}

for epsilon in epsilon_values:
    epsilon_greedy_average_rewards[epsilon] = run_bandit_experiment(epsilon=epsilon)
for value in greedy_optimistic_initial_values:
    greedy_optimistic_average_rewards[value] = run_bandit_experiment(initial_estimate=value, step_size=0.1)
for c in ucb_c_values:
    ucb_average_rewards[c] = run_bandit_experiment_ucb(c=c)
for step_size in gradient_bandit_step_sizes:
    gradient_bandit_average_rewards[step_size] = run_bandit_experiment_grad(step_size=step_size)

plt.plot(epsilon_greedy_average_rewards.keys(), epsilon_greedy_average_rewards.values(), label=f'epsilon-greedy')
plt.plot(greedy_optimistic_average_rewards.keys(), greedy_optimistic_average_rewards.values(), label=f'greedy optimistic initialization')
plt.plot(ucb_average_rewards.keys(), ucb_average_rewards.values(), label=f'upper confidence bound')
plt.plot(gradient_bandit_average_rewards.keys(), gradient_bandit_average_rewards.values(), label=f'gradient bandit')
plt.xlabel('parameter')
plt.xscale('log', base=2)
plt.ylabel('average reward over first 1000 time steps')
plt.title(f'parameter study, average reward')
plt.legend()
plt.show()