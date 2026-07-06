# RL-Based-Stock-Trading-Simulator:

A Python project that simulates stock price paths using Geometric Brownian Motion and applies a simple Q-learning agent to adjust portfolio stock allocation over time.

# Overview

This project combines:

- Monte Carlo simulation for generating stock price paths.
- Reinforcement learning using a Q-table and epsilon-greedy strategy.
- Portfolio rebalancing between stock and cash based on simulated market movements.
- Performance comparison between the learned RL strategy and a basic buy-and-hold approach.

The goal is to explore how a reinforcement learning agent can adapt portfolio allocation in a simulated market environment.

# Features

- Simulates stock prices using Geometric Brownian Motion.
- Represents portfolio allocation as discrete states.
- Uses Q-learning to learn action values.
- Supports three actions:
  - Decrease stock exposure
  - Hold position
  - Increase stock exposure
- Tracks:
  - Total reward per episode
  - Final portfolio value per episode
  - Learned Q-table
- Compares RL-driven strategy with buy-and-hold on a test price path.

# Tech Stack

- Python
- NumPy
- Matplotlib

# How It Works

1. A stock price path is generated using Monte Carlo simulation.
2. The portfolio starts with a fixed value and an initial 50/50 stock-cash allocation.
3. At each step, the agent observes the current stock weight state.
4. The agent selects an action using epsilon-greedy exploration.
5. The portfolio is updated based on the chosen action and simulated price movement.
6. Rewards are calculated from portfolio value changes.
7. The Q-table is updated using the Q-learning rule.
8. After training, the learned policy is evaluated against a buy-and-hold strategy.

# Project Structure

├── MCsimulator.py
└── README.md

# Requirements

Make sure you have Python 3 installed, along with:

```bash
pip install numpy matplotlib
```

# Usage

Run the script directly:

```bash
python MCsimulator.py
```

# Parameters

The main simulation uses values such as:

- Initial stock price: `100`
- Expected annual return: `0.08`
- Annual volatility: `0.20`
- Time horizon: `1 year`
- Time step: `1/252`
- Training episodes: `5000`

You can modify these parameters in the script to test different market conditions.

# Output

The script produces:

- A sample stock price simulation plot
- Training progress charts
- Final learned Q-table
- RL vs buy-and-hold comparison

# Example Use Case

This project can be used to study:

- Portfolio allocation under uncertainty
- Reinforcement learning in financial markets
- Dynamic rebalancing strategies
- Basic quantitative finance experimentation

# Notes

- This project is for educational and research purposes.
- Results depend heavily on simulation assumptions and hyperparameters.
- The strategy is trained on simulated data, so real-market performance may differ significantly.

