# ==============================================================================
# Step 1: Setup and Imports
# ==============================================================================
# No special libraries needed, just the basics!
import numpy as np
import matplotlib.pyplot as plt

# Let's make our plots look a bit nicer
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ Setup Complete!")

# ==============================================================================
# Step 2: The Environment - Simulating the Stock Market 📈
# ==============================================================================
# We'll use a famous model called Geometric Brownian Motion (GBM) to simulate
# the random "walk" of a stock price. It's the foundation of many financial models.

def simulate_stock_price(S0, mu, sigma, T, dt):
    """
    Simulates a stock price path using Geometric Brownian Motion.

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected annual return (the "drift").
        sigma (float): Annual volatility (the "randomness").
        T (int): Time horizon in years.
        dt (float): Time step size (e.g., 1/252 for daily steps).

    Returns:
        numpy.ndarray: An array of simulated stock prices.
    """
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)
    # The random part of the model: a standard normal distribution
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # This creates the Brownian motion path
    
    # The GBM formula
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)
    
    return S

# --- Let's test our simulation ---
# Parameters for the simulation
S0_test = 100        # Starting price of our stock
mu_test = 0.08       # Expected annual return of 8%
sigma_test = 0.20    # Annual volatility of 20% (typical for stocks)
T_test = 1           # Simulate for 1 year
dt_test = 1/252      # Daily steps (252 trading days in a year)

# Generate one path
price_path = simulate_stock_price(S0_test, mu_test, sigma_test, T_test, dt_test)

# Plot it to see what it looks like
plt.figure(figsize=(10, 5))
plt.title("Sample Monte Carlo Simulation of a Stock Price")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Stock Price ($)")
plt.plot(price_path)
plt.show()

# ==============================================================================
# Step 3: The Brain - Building the Q-Learning Agent 🧠
# ==============================================================================
# Q-learning is a simple form of RL. The agent learns a "Q-table" which tells
# it the expected future reward for taking a certain action in a certain state.
# Q(state, action) -> tells us how "good" it is to take 'action' when in 'state'.

# --- Define the State, Actions, and Q-Table ---

# 1. ACTIONS: What can our agent do?
# We'll keep it simple: decrease, hold, or increase stock allocation.
# Action 0: Decrease stock weight by 10% (and increase cash by 10%)
# Action 1: Hold current weights
# Action 2: Increase stock weight by 10% (and decrease cash by 10%)
ACTIONS = [0, 1, 2]
ACTION_EFFECT = -0.10 # The amount to change weight by for actions 0 and 2

# 2. STATE: How does the agent see the environment?
# The only thing our agent needs to know is its current portfolio allocation.
# We'll discretize the state. For example, if the agent has 65% in stocks,
# we can put it in the "60-70%" state bucket.
# Let's create 11 state buckets for stock weights from 0% to 100%.
# State 0: 0-9% stock | State 1: 10-19% stock | ... | State 10: 100% stock
NUM_STATES = 11

def get_state(stock_weight):
    """Converts a continuous stock weight (0.0 to 1.0) to a discrete state (0 to 10)."""
    # Ensure weight is within bounds [0, 1]
    stock_weight = np.clip(stock_weight, 0, 1)
    # Calculate the state index
    state = int(stock_weight * 10)
    return state

# 3. Q-TABLE: The agent's "cheat sheet" or brain.
# It's a table with rows for states and columns for actions.
# We initialize it with all zeros.
# Dimensions: (number of states) x (number of actions)
q_table = np.zeros((NUM_STATES, len(ACTIONS)))

print("Q-Table Initialized. Shape:", q_table.shape)
print("Initial Q-Table:")
print(q_table)

# ==============================================================================
# Step 4: The Training Loop - Learning from Experience
# ==============================================================================
# The agent will now "live" through many simulated years (episodes). In each
# episode, it will make decisions, see the results, and update its Q-table
# to get smarter.

# --- Hyperparameters for Q-Learning ---
LEARNING_RATE = 0.1  # alpha: How much we update our Q-values based on new info.
DISCOUNT_FACTOR = 0.99 # gamma: How much we value future rewards over immediate rewards.
EPISODES = 5000      # How many simulated years to train for.

# Epsilon for the "epsilon-greedy" strategy.
# It's the probability of choosing a random action (exploring) instead of the best one (exploiting).
# We start with a high epsilon and slowly decrease it.
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay_rate = 0.001

# --- Store results for plotting ---
rewards_per_episode = []
final_portfolio_values = []

print("\n🚀 Starting Training...")

# The main training loop
for episode in range(EPISODES):
    # --- Reset the environment for a new episode ---
    initial_portfolio_value = 10000  # Start with $10,000
    stock_weight = 0.50              # Start with a 50/50 split
    
    portfolio_value = initial_portfolio_value
    
    # Generate a new stock price path for this episode
    stock_prices = simulate_stock_price(S0_test, mu_test, sigma_test, T_test, dt_test)
    
    total_reward = 0
    
    # Loop through each day (time step) in the simulation
    for i in range(len(stock_prices) - 1):
        # 1. DETERMINE CURRENT STATE
        current_state = get_state(stock_weight)
        
        # 2. CHOOSE AN ACTION (Epsilon-Greedy)
        if np.random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action = np.random.choice(ACTIONS)
        else:
            # Exploit: choose the best action from the Q-table
            action = np.argmax(q_table[current_state, :])
            
        # 3. TAKE THE ACTION & CALCULATE REWARD
        # Get the value before the market moves
        old_portfolio_value = portfolio_value

        # Apply the action to change the weight
        if action == 0: # Decrease stock weight
            stock_weight += ACTION_EFFECT
        elif action == 2: # Increase stock weight
            stock_weight -= ACTION_EFFECT # Note: we subtract because ACTION_EFFECT is negative
        # For action 1 (Hold), do nothing.

        # Ensure weights are valid (between 0 and 1)
        stock_weight = np.clip(stock_weight, 0, 1)

        # Calculate the new portfolio value after the market moves
        # The change in value is based on the % change of the stock price,
        # weighted by how much we have in the stock.
        price_change_percent = (stock_prices[i+1] - stock_prices[i]) / stock_prices[i]
        portfolio_return = stock_weight * price_change_percent
        new_portfolio_value = old_portfolio_value * (1 + portfolio_return)

        # REWARD is the change in portfolio value
        reward = new_portfolio_value - old_portfolio_value
        total_reward += reward
        
        # Update portfolio value for the next step
        portfolio_value = new_portfolio_value

        # 4. GET NEW STATE & UPDATE Q-TABLE
        new_state = get_state(stock_weight)
        
        # The Q-Learning update rule
        old_q_value = q_table[current_state, action]
        next_max_q = np.max(q_table[new_state, :])
        
        new_q_value = (1 - LEARNING_RATE) * old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q)
        q_table[current_state, action] = new_q_value

    # --- End of Episode ---
    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
    
    # Log results
    rewards_per_episode.append(total_reward)
    final_portfolio_values.append(portfolio_value)
    
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}/{EPISODES} | Final Value: ${portfolio_value:,.2f} | Epsilon: {epsilon:.4f}")

print("\n✅ Training Complete!")

# ==============================================================================
# Step 5: Results and Evaluation 📊
# ==============================================================================

# --- Show the learned Q-Table ---
print("\n--- Final Learned Q-Table ---")
print("States (Stock Weight %): 0-9, 10-19, ..., 100")
print("Actions: 0 (DECREASE), 1 (HOLD), 2 (INCREASE)")
print(q_table.round(2))
# Interpretation: A high positive value means the agent expects a good reward
# for taking that action in that state.

# --- Plot the training progress ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward ($)")

plt.subplot(1, 2, 2)
plt.plot(final_portfolio_values)
plt.title("Final Portfolio Value per Episode")
plt.xlabel("Episode")
plt.ylabel("Final Value ($)")
plt.tight_layout()
plt.show()

# --- Compare the learned policy against a simple strategy ---
print("\n--- Comparing RL Agent vs. Buy-and-Hold Strategy ---")

# Let's run a final test on a new, unseen stock path
test_prices = simulate_stock_price(S0_test, mu_test, sigma_test, T_test, dt_test)

# --- Strategy 1: The RL Agent ---
rl_portfolio_values = []
portfolio_value = 10000
stock_weight = 0.50 # Start at 50/50
rl_weights = []

for i in range(len(test_prices) - 1):
    rl_portfolio_values.append(portfolio_value)
    rl_weights.append(stock_weight)
    
    current_state = get_state(stock_weight)
    # Use the learned policy (always choose the best action)
    action = np.argmax(q_table[current_state, :])
    
    if action == 0:
        stock_weight += ACTION_EFFECT
    elif action == 2:
        stock_weight -= ACTION_EFFECT
    
    stock_weight = np.clip(stock_weight, 0, 1)
    
    price_change_percent = (test_prices[i+1] - test_prices[i]) / test_prices[i]
    portfolio_return = stock_weight * price_change_percent
    portfolio_value *= (1 + portfolio_return)

# --- Strategy 2: Buy and Hold (50/50 split) ---
buy_hold_values = []
portfolio_value = 10000
stock_weight = 0.50 # Fixed weight

for i in range(len(test_prices) - 1):
    buy_hold_values.append(portfolio_value)
    price_change_percent = (test_prices[i+1] - test_prices[i]) / test_prices[i]
    portfolio_return = stock_weight * price_change_percent
    portfolio_value *= (1 + portfolio_return)

# --- Plot the comparison ---
plt.figure(figsize=(12, 8))

# Portfolio Value Plot
plt.subplot(2, 1, 1)
plt.title("Performance Comparison: RL Agent vs. Buy-and-Hold")
plt.plot(rl_portfolio_values, label=f"RL Agent (Final: ${rl_portfolio_values[-1]:,.2f})", color='blue')
plt.plot(buy_hold_values, label=f"Buy & Hold 50/50 (Final: ${buy_hold_values[-1]:,.2f})", color='orange', linestyle='--')
plt.ylabel("Portfolio Value ($)")
plt.legend()

# Agent's Weight Allocation Plot
plt.subplot(2, 1, 2)
plt.plot(rl_weights, label="RL Agent's Stock Weight", color='blue', alpha=0.7)
plt.axhline(y=0.5, color='orange', linestyle='--', label='Buy & Hold Weight (50%)')
plt.title("Agent's Stock Allocation Over Time")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Stock Weight %")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()