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
    
class GradientAgent:
    def __init__(self, actions, 
                 alpha, # step size or learning rate
                 include_baseline=True):
        self.actions = actions
        self.alpha = alpha
        self.include_baseline = include_baseline

        self.estimated_mean = 0
        self.action_counts = np.zeros(actions)
        self.t = 0 # internal time
        self.H = np.zeros(actions) # preference vector
        self.dist = np.ones(actions) / actions

    def action(self):
        self.H = self.H - np.max(self.H)
        dist = np.exp(self.H)
        dist = dist / np.sum(dist)
        action = np.random.choice(range(self.actions), p=dist)
        self.most_recent_action = action
        self.dist = dist
        return action

    def update(self, reward, action=None):
        if action is None:
            action = self.most_recent_action
        self.t += 1
        self.action_counts[action] += 1
        self.estimated_mean += (reward - self.estimated_mean) / self.t
        action_vec = np.eye(self.actions)[action]
        self.H += self.alpha * (reward - self.include_baseline * self.estimated_mean) * (action_vec - self.dist)

    def reset(self):
        self.estimated_mean = 0
        self.action_counts = np.zeros(self.actions)
        self.t = 0
        self.H = np.zeros(self.actions)
        self.dist = np.ones(self.actions) / self.actions
