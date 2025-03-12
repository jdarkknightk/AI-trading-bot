import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces
from stable_baselines3 import DQN

# Function to compute RSI
def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.data['rsi'] = compute_rsi(self.data['aapl.close'])
        self.data['sma'] = self.data['aapl.close'].rolling(50).mean()
        self.data.fillna(0, inplace=True)  # Handle NaN values
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_profit = 0
        return self._next_observation(), {}
    
    def _next_observation(self):
        obs = np.array([
            self.balance,
            self.shares_held,
            self.data.iloc[self.current_step]['aapl.open'],
            self.data.iloc[self.current_step]['aapl.high'],
            self.data.iloc[self.current_step]['aapl.low'],
            self.data.iloc[self.current_step]['aapl.close'],
            self.data.iloc[self.current_step]['rsi'],
            self.data.iloc[self.current_step]['sma'],
        ])
        return obs
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['aapl.close']
        reward = 0
        done = False
        
        if action == 1:  # Buy
            if self.balance >= current_price:
                shares_bought = self.balance // current_price
                self.shares_held += shares_bought
                self.balance -= shares_bought * current_price
                self.total_shares_bought += shares_bought
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.total_shares_sold += self.shares_held
                self.shares_held = 0
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        # Improved reward function
        total_asset_value = self.balance + (self.shares_held * current_price)
        reward = total_asset_value - self.initial_balance
        
        return self._next_observation(), reward, done, False, {}

# Load Stock Data
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
print("Dataset Columns:", data.columns)

env = StockTradingEnv(data)

# Train the RL Model
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)  # Increased training time

# Test the Model
done = False
obs, _ = env.reset()
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

print("Final Balance:", env.balance)
