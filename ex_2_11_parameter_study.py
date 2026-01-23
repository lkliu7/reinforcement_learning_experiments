import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from bandit import Bandit
from greedy_agent import GreedyAgent
from ucb_agent import UCBAgent
from gradient_agent import GradientAgent

CONFIG = {
    'n_bandits': 10,
    'warm_up': 0,
    'time_steps': 1000,
    'runs': 10000,
    'epsilon_values': [1/128, 1/64, 1/32, 1/16, 1/8, 1/4],
    'gradient_bandit_step_sizes': [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    'ucb_c_values': [1/16, 1/8, 1/4, 1/2, 1, 2, 4],
    'greedy_optimistic_initial_values': [1/4, 1/2, 1, 2, 4],
    'progress_interval': 500,
}

n_bandits = CONFIG['n_bandits']
time_steps = CONFIG['time_steps']
runs = CONFIG['runs']
epsilon_values = CONFIG['epsilon_values']
greedy_optimistic_initial_values = CONFIG['greedy_optimistic_initial_values']
gradient_bandit_step_sizes = CONFIG['gradient_bandit_step_sizes']
ucb_c_values = CONFIG['ucb_c_values']
progress_interval = CONFIG['progress_interval']
        
def run_bandit_experiment(agent, bandit, warm_up = 0,
                          time_steps = time_steps, runs = runs,
                          progress_interval = progress_interval):
    """Run bandit experiment across multiple independent runs and return average reward."""
    average_rewards_across_runs = 0

    for run in range(runs):
        agent.reset()
        bandit.reset()
        total_rewards = 0

        # Warm-up phase: agent learns but rewards don't count toward performance
        for t in range(warm_up):
            action = agent.action()
            reward = bandit.pull(action)
            agent.update(reward, action)

        for t in range(time_steps):
            action = agent.action()
            reward = bandit.pull(action)
            agent.update(reward, action)
            total_rewards += reward

        average_rewards_across_runs += total_rewards / time_steps

        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    average_rewards_across_runs /= runs
    return average_rewards_across_runs

def call_run_bandit_experiment(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    return run_bandit_experiment(*args)

def distribute_bandit_experiment(agent, bandit, cores = os.cpu_count(), warm_up = 0,
                          time_steps = time_steps, runs = runs,
                          progress_interval = progress_interval):
    """Distribute experiment runs across multiple CPU cores for parallel execution."""
    # Split total runs across available cores
    counts = [runs // cores] * cores
    for i in range(runs % cores):
        counts[i] += 1
    # Create argument tuples for each process (deepcopy ensures independence)
    args_list = [(deepcopy(agent), deepcopy(bandit), warm_up, time_steps, count, progress_interval) for count in counts if count > 0]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(call_run_bandit_experiment, args_list))
    # Weighted average: each process contributes proportionally to runs completed
    return sum([results[i] * counts[i] for i in range(len(results))]) / runs

if __name__ == '__main__':
    # Replicate Sutton and Barto, Figure 2.6
    # Run parameter sweeps for each algorithm
    epsilon_greedy_average_rewards = {}
    greedy_optimistic_average_rewards = {}
    ucb_average_rewards = {}
    gradient_bandit_average_rewards = {}

    # Stationary bandit environment for first experiment
    bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=1, mean_drift_std=0)

    for epsilon in epsilon_values:
        print(f'epsilon = {epsilon}')
        agent = GreedyAgent(actions=n_bandits, epsilon=epsilon)
        epsilon_greedy_average_rewards[epsilon] = distribute_bandit_experiment(agent=agent, bandit=bandit)

    for value in greedy_optimistic_initial_values:
        print(f'initialization = {value}')
        agent = GreedyAgent(actions=n_bandits, initial_estimate=value, step_size=0.1)
        greedy_optimistic_average_rewards[value] = distribute_bandit_experiment(agent=agent, bandit=bandit)

    for c in ucb_c_values:
        print(f'c = {c}')
        agent = UCBAgent(actions=n_bandits, c=c)
        ucb_average_rewards[c] = distribute_bandit_experiment(agent=agent, bandit=bandit)

    for alpha in gradient_bandit_step_sizes:
        print(f'alpha = {alpha}')
        agent = GradientAgent(actions=n_bandits, alpha=alpha)
        gradient_bandit_average_rewards[alpha] = distribute_bandit_experiment(agent=agent, bandit=bandit)

    plt.plot(epsilon_greedy_average_rewards.keys(), epsilon_greedy_average_rewards.values(), label=f'epsilon-greedy')
    plt.plot(greedy_optimistic_average_rewards.keys(), greedy_optimistic_average_rewards.values(), label=f'greedy optimistic initialization')
    plt.plot(ucb_average_rewards.keys(), ucb_average_rewards.values(), label=f'upper confidence bound')
    plt.plot(gradient_bandit_average_rewards.keys(), gradient_bandit_average_rewards.values(), label=f'gradient bandit')
    plt.xlabel('parameter')
    plt.xscale('log', base=2)  # Log scale to better visualize wide parameter ranges
    plt.ylabel('average reward over first 1000 time steps')
    plt.title(f'parameter study, average reward')
    plt.legend()
    plt.show()

    # Sutton and Barto, Exercise 2.11: Nonstationary bandit comparison
    # Reset result dictionaries for second experiment
    epsilon_greedy_average_rewards = {}
    epsilon_greedy_constant_step_average_rewards = {}
    greedy_optimistic_average_rewards = {}
    ucb_average_rewards = {}
    ucb_constant_step_average_rewards = {}
    gradient_bandit_average_rewards = {}

    # Nonstationary bandit: means start at 0 and undergo random walk
    bandit = Bandit(n_bandits=n_bandits, mean_baseline=0, mean_std=0, mean_drift_std=0.01)

    for epsilon in epsilon_values:
        print(f'epsilon = {epsilon}')
        agent = GreedyAgent(actions=n_bandits, epsilon=epsilon)
        epsilon_greedy_average_rewards[epsilon] = distribute_bandit_experiment(agent=agent, bandit=bandit, warm_up=100000, time_steps=100000)

    for epsilon in epsilon_values:
        print(f'epsilon = {epsilon}')
        # Constant step size for better tracking in nonstationary environment
        agent = GreedyAgent(actions=n_bandits, epsilon=epsilon, step_size=0.1)
        epsilon_greedy_constant_step_average_rewards[epsilon] = distribute_bandit_experiment(agent=agent, bandit=bandit, warm_up=100000, time_steps=100000)

    for value in greedy_optimistic_initial_values:
        print(f'initialization = {value}')
        agent = GreedyAgent(actions=n_bandits, initial_estimate=value, step_size=0.1)
        greedy_optimistic_average_rewards[value] = distribute_bandit_experiment(agent=agent, bandit=bandit, warm_up=100000, time_steps=100000)

    for c in ucb_c_values:
        print(f'c = {c}')
        agent = UCBAgent(actions=n_bandits, c=c)
        ucb_average_rewards[c] = distribute_bandit_experiment(agent=agent, bandit=bandit, warm_up=100000, time_steps=100000)

    for c in ucb_c_values:
        print(f'c = {c}')
        # UCB with constant step size for nonstationary environment
        agent = UCBAgent(actions=n_bandits, c=c, step_size=0.1)
        ucb_constant_step_average_rewards[c] = distribute_bandit_experiment(agent=agent, bandit=bandit, warm_up=100000, time_steps=100000)

    for alpha in gradient_bandit_step_sizes:
        print(f'alpha = {alpha}')
        agent = GradientAgent(actions=n_bandits, alpha=alpha)
        gradient_bandit_average_rewards[alpha] = distribute_bandit_experiment(agent=agent, bandit=bandit, warm_up=100000, time_steps=100000)

    plt.plot(epsilon_greedy_average_rewards.keys(), epsilon_greedy_average_rewards.values(), label=f'epsilon-greedy')
    plt.plot(epsilon_greedy_constant_step_average_rewards.keys(), epsilon_greedy_constant_step_average_rewards.values(), label=f'epsilon-greedy exponential average')
    plt.plot(greedy_optimistic_average_rewards.keys(), greedy_optimistic_average_rewards.values(), label=f'greedy optimistic initialization')
    plt.plot(ucb_average_rewards.keys(), ucb_average_rewards.values(), label=f'upper confidence bound')
    plt.plot(ucb_constant_step_average_rewards.keys(), ucb_constant_step_average_rewards.values(), label=f'upper confidence bound exponential average')
    plt.plot(gradient_bandit_average_rewards.keys(), gradient_bandit_average_rewards.values(), label=f'gradient bandit')
    plt.xlabel('parameter')
    plt.xscale('log', base=2)  # Log scale to better visualize wide parameter ranges
    plt.ylabel('average reward over last 100000 time steps')
    plt.title('Nonstationary Bandit Parameter Study')
    plt.legend()
    plt.show()