# Multi-Armed Bandit Experiments

This repository contains implementations of classic multi-armed bandit algorithms and experiments from Sutton & Barto's "Reinforcement Learning: An Introduction".

## Repository Structure

### Core Classes
- **`bandit.py`** - Multi-armed bandit environment with support for stationary and nonstationary reward distributions
- **`greedy_agent.py`** - Epsilon-greedy agent with configurable exploration rate and step sizes
- **`ucb_agent.py`** - Upper Confidence Bound (UCB) agent for optimistic action selection
- **`gradient_agent.py`** - Gradient bandit agent using softmax action selection with preference learning

### Experiment Framework
- **`experiment_utils.py`** - Shared utilities for running experiments and plotting results across multiple agents and configurations

### Classic Experiments (Sutton & Barto Replications)

#### Stationary Bandits
- **`fig_2_2_epsilon_greedy.py`** - Replicates Figure 2.2: 10-armed testbed comparing epsilon-greedy methods with different exploration rates (ε = 0, 0.01, 0.1)

- **`fig_2_4_ucb.py`** - Replicates Figure 2.4: Upper-Confidence-Bound action selection compared to epsilon-greedy

- **`fig_2_5_gradient.py`** - Replicates Figure 2.5: Gradient bandit algorithm comparing different step sizes and baseline effects

#### Nonstationary Bandits
- **`ex_2_5_nonstationary.py`** - Implements Exercise 2.5: Compares sample averaging vs exponential recency-weighted averaging in nonstationary environments

#### Parameter Studies
- **`ex_2_11_parameter_study.py`** - Comprehensive parameter sweep study comparing all algorithms across different configurations, with parallel processing support

### Other Files
- **`stationary_bandits.wls`** - Wolfram Language implementation of epsilon-greedy experiments
- **`nonstationary_bandits.wls`** - Wolfram Language implementation of nonstationary bandit experiments

## Key Features

### Algorithm Implementations
- **Epsilon-Greedy**: Configurable exploration rate, optimistic initialization, sample averaging or constant step sizes
- **Upper Confidence Bound (UCB)**: Optimism in the face of uncertainty with configurable confidence parameter
- **Gradient Bandit**: Preference-based learning with optional baseline subtraction and numerical stability

### Environment Support
- **Stationary Bandits**: Fixed reward distributions
- **Nonstationary Bandits**: Random walk reward distributions for studying adaptation
- **Configurable Parameters**: Number of arms, reward variance, baseline shifts, drift rates

### Experiment Infrastructure
- **Parallel Processing**: Multi-core experiment execution for faster parameter studies
- **Standardized Interface**: Consistent experiment running and result collection
- **Visualization**: Automated plotting of average rewards and optimal action percentages
- **Reproducibility**: Configurable random seeds and experiment parameters

## Usage

Each experiment file can be run independently:

```bash
# Compare epsilon-greedy exploration rates (Figure 2.2)
python fig_2_2_epsilon_greedy.py

# Compare UCB vs epsilon-greedy (Figure 2.4)
python fig_2_4_ucb.py

# Compare gradient bandit with/without baseline (Figure 2.5)
python fig_2_5_gradient.py

# Study nonstationary adaptation (Exercise 2.5)
python ex_2_5_nonstationary.py

# Comprehensive parameter study (Exercise 2.11)
python ex_2_11_parameter_study.py
```

## Design Principles

- **Modular Architecture**: Separate environment, agent, and experiment concerns
- **Academic Fidelity**: Faithful implementation of textbook algorithms and experiments
- **Performance**: Optimized for large-scale parameter studies with multiprocessing