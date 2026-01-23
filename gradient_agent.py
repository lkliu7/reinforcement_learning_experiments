import numpy as np
    
class GradientAgent:
    """Gradient bandit agent using softmax action probabilities.

    Maintains preference values (H) for each action and uses gradient ascent
    to update preferences based on rewards relative to a baseline.
    """
    def __init__(self, actions, 
                 alpha, # step size or learning rate
                 include_baseline=True):
        self.actions = actions
        self.alpha = alpha
        self.include_baseline = include_baseline

        self.estimated_mean = 0
        self.action_counts = np.zeros(actions)
        self.t = 0  # Internal time step counter
        self.H = np.zeros(actions)  # Action preferences (logits)

    def action(self):
        action = np.random.choice(range(self.actions), p=self.dist)
        self.most_recent_action = action
        return action

    def update(self, reward, action=None):
        if action is None:
            action = self.most_recent_action
        self.t += 1
        self.action_counts[action] += 1
        # Update running average of all rewards (baseline)
        self.estimated_mean += (reward - self.estimated_mean) / self.t
        # One-hot encoding of selected action
        action_vec = np.zeros(self.actions)
        action_vec[action] = 1
        # Gradient ascent on preferences: increase H for chosen action if reward > baseline
        self.H += self.alpha * (reward - self.include_baseline * self.estimated_mean) * (action_vec - self.dist)
        # Normalize to keep max preference at 0 for numerical stability
        self.H = self.H - np.max(self.H)

    def reset(self):
        self.estimated_mean = 0
        self.action_counts = np.zeros(self.actions)
        self.t = 0
        self.H = np.zeros(self.actions)

    @property
    def dist(self):
        """Compute softmax probabilities from current preferences."""
        softmax = np.exp(self.H)
        return softmax / np.sum(softmax)
