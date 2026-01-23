import numpy as np
import matplotlib.pyplot as plt

def run_experiment(bandit, agent, time_steps, runs, progress_interval=100):
    """Run a bandit experiment with the given bandit and agent.

    Args:
        bandit: Bandit environment instance
        agent: Agent instance (must have action() and update() methods)
        time_steps: Number of steps per run
        runs: Number of independent runs
        progress_interval: How often to print progress

    Returns:
        tuple: (average_rewards, optimal_action_frequencies)
    """
    total_rewards = np.zeros(time_steps)
    total_optimal_actions = np.zeros(time_steps)

    for run in range(runs):
        # Reset environment and agent for each run
        bandit.reset()
        agent.reset()

        rewards = []
        optimal_actions = []

        for t in range(time_steps):
            # Agent selects action
            action = agent.action()

            # Environment provides reward
            reward = bandit.pull(action)

            # Agent updates estimates
            agent.update(reward)

            # Track performance
            rewards.append(reward)
            optimal_actions.append(1 if action == bandit.optimal_arm else 0)

        total_rewards += np.array(rewards)
        total_optimal_actions += np.array(optimal_actions)

        if run % progress_interval == progress_interval - 1:
            print(f'Completed run {run + 1}.')

    # Average across all runs
    average_rewards = total_rewards / runs
    optimal_action_frequencies = total_optimal_actions / runs

    return average_rewards, optimal_action_frequencies

def plot_performance(results_dict, title_prefix="", ylabel_rewards="average reward",
                    ylabel_optimal="optimal action ratio"):
    """Plot performance comparison across different agents/parameters.

    Args:
        results_dict: Dict with (label -> (rewards, optimal_actions)) mapping
        title_prefix: Prefix for plot titles
        ylabel_rewards: Y-axis label for rewards plot
        ylabel_optimal: Y-axis label for optimal actions plot
    """
    time_steps = len(next(iter(results_dict.values()))[0])

    # Plot average rewards
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label, (rewards, _) in results_dict.items():
        plt.plot(range(1, time_steps + 1), rewards, label=label)
    plt.xlabel('steps')
    plt.ylabel(ylabel_rewards)
    plt.title(f'{title_prefix} Average Rewards' if title_prefix else 'Average Rewards')
    plt.legend()

    # Plot optimal action frequency
    plt.subplot(1, 2, 2)
    for label, (_, optimal_actions) in results_dict.items():
        plt.plot(range(1, time_steps + 1), optimal_actions, label=label)
    plt.xlabel('steps')
    plt.ylabel(ylabel_optimal)
    plt.title(f'{title_prefix} Optimal Action %' if title_prefix else 'Optimal Action %')
    plt.legend()

    plt.tight_layout()
    plt.show()