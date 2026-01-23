import random
import numpy as np

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
    
def sample_average_step_size(n):
    return 1 / n
    
class GreedyAgent:
    """Epsilon-greedy bandit agent that balances exploration and exploitation.

    With probability epsilon, selects a random action (exploration).
    Otherwise, selects the action with highest estimated value (exploitation).
    """
    def __init__(self, actions, epsilon=0, initial_estimate=0, step_size=sample_average_step_size):
        self.epsilon = epsilon
        self.initial_estimate = initial_estimate
        self.actions = actions
        self.step_size = step_size

        self.estimated_means = np.zeros(actions) + initial_estimate
        self.action_counts = np.zeros(actions)

    def action(self):
        greedy_action = self.estimated_means.argmax()
        # Epsilon-greedy action selection: explore with probability epsilon
        if random.random() < self.epsilon:
            action = np.random.randint(self.actions)  # Random exploration
        else:
            action = greedy_action  # Greedy exploitation
        self.most_recent_action = action
        return action
    
    def update(self, reward, action=None):
        if action is None:
            action = self.most_recent_action
        self.action_counts[action] += 1
        # Update action value estimate using incremental average
        self.estimated_means[action] += (reward - self.estimated_means[action]) * get_step_size(self.step_size, self.action_counts[action])

    def reset(self):
        self.estimated_means = np.zeros(self.actions) + self.initial_estimate
        self.action_counts = np.zeros(self.actions)
