import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from collections import deque

# --- Configuration ---
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
GAMMA = 0.99  # Discount factor for future rewards
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000
BATCH_SIZE = 32

# --- 1. The Environment (Simulation) ---
class TradingEnv:
    def __init__(self, prices, headlines):
        self.prices = prices
        self.headlines = headlines
        self.n_step = 0
        self.cash = 10000
        self.holdings = 0
        self.initial_balance = 10000
        
    def reset(self):
        self.n_step = 0
        self.cash = 10000
        self.holdings = 0
        return self._get_state()
        
    def _get_state(self):
        # State = [Current Price, Cash, Holdings, Headline_Index]
        # In a real scenario, we wouldn't include the index, but the embedding itself comes later
        return {
            "price": self.prices[self.n_step],
            "cash": self.cash,
            "holdings": self.holdings,
            "headline": self.headlines[self.n_step]
        }
    
    def step(self, action):
        # Actions: 0=Hold, 1=Buy, 2=Sell
        current_price = self.prices[self.n_step]
        reward = 0
        
        if action == 1: # Buy
            if self.cash >= current_price:
                self.cash -= current_price
                self.holdings += 1
                # Reward is slightly negative for transaction cost usually, 
                # but here we reward "potential" if price goes up next
                pass
        elif action == 2: # Sell
            if self.holdings > 0:
                self.cash += current_price
                self.holdings -= 1
                # Realized profit calculation could go here
                
        # Move to next day
        self.n_step += 1
        done = self.n_step >= len(self.prices) - 1
        
        # Calculate Total Portfolio Value for Reward
        next_price = self.prices[self.n_step] if not done else current_price
        portfolio_value = self.cash + (self.holdings * next_price)
        
        # Simple Reward: Change in portfolio value
        prev_value = self.cash + (self.holdings * current_price) # Note: cash/holdings already updated
        # To get true prev value we'd need to track it before action, but this is a simple proxy
        # Let's just use: (New Value - Initial Value) as a sparse reward or daily return
        reward = portfolio_value - 10000 # Reward is total profit so far
        
        return self._get_state(), reward, done

# --- 2. The Hybrid Agent (DeepSeek + DQN) ---
class DeepSeekDQN(nn.Module):
    def __init__(self, base_model, hidden_size=2048):
        super().__init__()
        self.base_model = base_model
        # Freeze DeepSeek
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # DQN Head: Takes [Embedding (2048) + Price (1) + Cash (1) + Holdings (1)]
        self.input_dim = hidden_size + 3 
        
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # Q-Values for [Hold, Buy, Sell]
        )
        
    def forward(self, input_ids, attention_mask, numeric_state):
        # 1. Get Text Embedding from DeepSeek
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Last hidden state of last token
            text_embedding = outputs.hidden_states[-1][:, -1, :] # [Batch, 2048]
            
        # 2. Concatenate with Numeric State (Price, Cash, Holdings)
        # numeric_state shape: [Batch, 3]
        combined_input = torch.cat((text_embedding.float(), numeric_state), dim=1)
        
        # 3. Predict Q-Values
        q_values = self.network(combined_input)
        return q_values

# --- 3. Setup & Training ---
print("Loading DeepSeek Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    torch_dtype=torch.float16
)

agent = DeepSeekDQN(base_model)
optimizer = optim.Adam(agent.network.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Dummy Data
# Scenario: Good news but price is dropping (Value trap?), Bad news but price rising?
# Let's make a simple trend: Price goes UP, then CRASHES.
prices = [100, 105, 110, 115, 120, 118, 115, 90, 85, 80]
headlines = [
    "Market looks stable.",
    "Company reports growth.",
    "Analysts predict bull run.",
    "Record breaking earnings.", # Peak price
    "Slight correction expected.",
    "Uncertainty in the market.",
    "Rumors of accounting fraud.", # Crash starts
    "CEO investigation confirmed.",
    "Massive sell-off continues.",
    "Company declares bankruptcy."
]

env = TradingEnv(prices, headlines)

print("\n--- Starting Training Loop ---")
# We will run just a few episodes to demonstrate the mechanics
for episode in range(5):
    state = env.reset()
    total_reward = 0
    done = False
    
    print(f"\nEpisode {episode+1}:")
    
    while not done:
        # Prepare inputs
        text_input = tokenizer(state["headline"], return_tensors="pt")
        numeric_input = torch.tensor([[
            state["price"], 
            state["cash"], 
            state["holdings"]
        ]], dtype=torch.float32)
        
        # Epsilon-Greedy Action
        if random.random() < EPSILON:
            action = random.randint(0, 2)
            action_type = "Random"
        else:
            with torch.no_grad():
                q_values = agent(text_input.input_ids, text_input.attention_mask, numeric_input)
                action = torch.argmax(q_values).item()
                action_type = "Model "
        
        # Take Action
        next_state, reward, done = env.step(action)
        
        # Print Step Info
        actions = ["HOLD", "BUY ", "SELL"]
        print(f"  Step {env.n_step-1}: Price={state['price']:.0f} | News='{state['headline'][:20]}...'")
        print(f"    -> Action: {actions[action]} ({action_type}) | Reward: {reward:.0f}")
        
        # Train (Simplified - normally we use a Replay Buffer)
        # Target = Reward + Gamma * max(Q(next_state))
        target = reward
        if not done:
            next_text = tokenizer(next_state["headline"], return_tensors="pt")
            next_numeric = torch.tensor([[next_state["price"], next_state["cash"], next_state["holdings"]]], dtype=torch.float32)
            with torch.no_grad():
                next_q = agent(next_text.input_ids, next_text.attention_mask, next_numeric)
                target = reward + GAMMA * torch.max(next_q).item()
        
        # Update
        current_q = agent(text_input.input_ids, text_input.attention_mask, numeric_input)
        target_vec = current_q.clone()
        target_vec[0][action] = target
        
        loss = loss_fn(current_q, target_vec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
        total_reward = reward # In this env, reward is cumulative profit
        
    # Decay Epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY
        
    print(f"  >> Episode Finished. Final Profit: ${total_reward:.2f}")

print("\n--- Comparison Analysis ---")
print("A Simple Classifier would likely predict 'BUY' for 'Record breaking earnings' (Step 3).")
print("However, a trained DQN might learn to 'SELL' there if it learns that price usually crashes after such peaks in this specific dataset.")
print("The DQN also considers 'Cash'. If Cash=0, the DQN knows it CANNOT Buy, whereas a Classifier would still output 'BUY' signal.")
