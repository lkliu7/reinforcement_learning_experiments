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
    
class UCBAgent:
    def __init__(self, actions, c=1, step_size=lambda n:1/n):
        self.actions = actions
        self.step_size = step_size
        self.c = c

        self.estimated_means = np.zeros(actions)
        self.action_counts = np.zeros(actions)
        self.t = 0

    def action(self):
        ucb_values = np.where(self.action_counts > 0, self.estimated_means + self.c * np.sqrt(np.log(self.t + 1) / self.action_counts), np.inf)
        action = ucb_values.argmax()
        self.most_recent_action = action
        return action

    def update(self, reward, action=None):
        if action is None:
            action = self.most_recent_action
        self.t += 1
        self.action_counts[action] += 1
        self.estimated_means[action] = self.estimated_means[action] + (reward - self.estimated_means[action]) * get_step_size(self.step_size, self.action_counts[action])

    def reset(self):
        self.estimated_means = np.zeros(self.actions)
        self.action_counts = np.zeros(self.actions)
        self.t = 0
