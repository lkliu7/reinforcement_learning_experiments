import random
import numpy as np

class Bandit:
    def __init__(self,
                 n_bandits = 1,
                 mean_baseline = 0,
                 mean_std = 1,
                 reward_std = 1,
                 mean_drift = 0):
        self.n_bandits = n_bandits
        self.mean_baseline = mean_baseline
        self.mean_std = mean_std
        self.reward_std = reward_std
        self.mean_drift = mean_drift
        self.bandit_means = self.mean_baseline + np.random.randn(self.n_bandits) * self.mean_std
    
    def pull(self, arm = None):
        if arm is None:
            arm = random.randint(0, self.n_bandits - 1)
        reward = self.bandit_means[arm] + np.random.randn() * self.reward_std
        if self.mean_drift != 0:
            self.bandit_means += np.random.randn(self.n_bandits) * self.mean_drift
        return reward
    
    @property
    def optimal_arm(self):
        return self.bandit_means.argmax()
    
    def reset(self):
        self.bandit_means = self.mean_baseline + np.random.randn(self.n_bandits) * self.mean_std
