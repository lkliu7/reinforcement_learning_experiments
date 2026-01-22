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
    
class GreedyAgent:
    def __init__(self, actions, epsilon=0, initial_estimate=0, step_size=lambda n:1/n):
        self.epsilon = epsilon
        self.initial_estimate = initial_estimate
        self.actions = actions
        self.step_size = step_size

        self.estimated_means = np.zeros(actions) + initial_estimate
        self.action_counts = np.zeros(actions)

    def action(self):
        greedy_action = self.estimated_means.argmax()
        if random.random() < self.epsilon:
            action = np.random.randint(self.actions)
        else:
            action = greedy_action
        self.most_recent_action = action
        return action
    
    def update(self, reward, action=None):
        if action is None:
            action = self.most_recent_action
        self.action_counts[action] += 1
        self.estimated_means[action] += (reward - self.estimated_means[action]) * get_step_size(self.step_size, self.action_counts[action])

    def reset(self):
        self.estimated_means = np.zeros(self.actions) + self.initial_estimate
        self.action_counts = np.zeros(self.actions)
