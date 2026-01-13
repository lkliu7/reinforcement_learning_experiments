# reinforcement_learning_experiments

## Stationary Multi-Armed Bandit Experiments

### Files

- **`stationary_bandits.py`** - Python implementation of the 10-armed testbed experiment from Sutton & Barto's "Reinforcement Learning: An Introduction". Compares epsilon-greedy action selection with different exploration rates (ε = 0, 0.01, 0.1) across 2000 runs of 1000 time steps each.

- **`stationary_bandits.wls`** - Mathematica/Wolfram Language implementation of the same experiment, providing identical functionality with Wolfram's statistical and visualization capabilities.

Both implementations replicate the classic experiments and plots found in the 10-armed testbed literature, demonstrating how different levels of exploration affect average reward and optimal action selection over time.