import matplotlib.pyplot as plt

from bandit import Bandit
from greedy_agent import GreedyAgent
from ucb_agent import UCBAgent
from gradient_agent import GradientAgent

CONFIG = {
    'n_bandits': 10,
    'time_steps': 1000,
    'runs': 2000,
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
        
def run_bandit_experiment(agent, n_bandits = n_bandits,
                          mean_drift = 0,  # 0 = stationary, >0 = nonstationary
                          time_steps = time_steps, runs = runs,
                          progress_interval = progress_interval):
    """Run bandit experiment across multiple independent runs and return average reward."""
    bandit = Bandit(n_bandits=n_bandits, mean_drift=mean_drift)
    average_rewards_across_runs = 0

    for run in range(runs):
        agent.reset()
        total_rewards = 0

        for t in range(time_steps):
            action = agent.action()
            reward = bandit.pull(action)
            agent.update(reward, action)
            total_rewards += reward

        bandit.reset()
        average_rewards_across_runs += total_rewards / time_steps

        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    average_rewards_across_runs /= runs
    return average_rewards_across_runs

# Run parameter sweeps for each algorithm
epsilon_greedy_average_rewards = {}
greedy_optimistic_average_rewards = {}
ucb_average_rewards = {}
gradient_bandit_average_rewards = {}

for epsilon in epsilon_values:
    print(f'epsilon = {epsilon}')
    agent = GreedyAgent(actions=n_bandits, epsilon=epsilon)
    epsilon_greedy_average_rewards[epsilon] = run_bandit_experiment(agent=agent)

for value in greedy_optimistic_initial_values:
    print(f'initialization = {value}')
    agent = GreedyAgent(actions=n_bandits, initial_estimate=value, step_size=0.1)
    greedy_optimistic_average_rewards[value] = run_bandit_experiment(agent=agent)

for c in ucb_c_values:
    print(f'c = {c}')
    agent = UCBAgent(actions=n_bandits, c=c)
    ucb_average_rewards[c] = run_bandit_experiment(agent=agent)

for alpha in gradient_bandit_step_sizes:
    print(f'alpha = {alpha}')
    agent = GradientAgent(actions=n_bandits, alpha=alpha)
    gradient_bandit_average_rewards[alpha] = run_bandit_experiment(agent=agent)

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