import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
import random
import time

# --- Configuration ---
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
DATA_FILE = "market_data.csv"
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 5 # Keep low for demo speed, increase for real results

# --- Data Loading & Preprocessing ---
def load_and_process_data(filepath):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Resample to Weekly (Friday) to reduce steps and noise
    # We take the last value of the week
    df_weekly = df.resample('W-FRI').last()
    
    # Recalculate Log Returns for Weekly
    df_weekly["QQQ_Log_Return"] = np.log(df_weekly["QQQ"] / df_weekly["QQQ"].shift(1))
    df_weekly = df_weekly.dropna()
    
    # Features to use
    feature_cols = ["QQQ_Log_Return", "VIX", "RSI", "MACD", "DGS2", "CPI_YoY", "Unemployment"]
    
    # Normalize (Simple Min-Max or Standardization)
    # For RL, Standardization (Mean 0, Std 1) is usually better
    data_mean = df_weekly[feature_cols].mean()
    data_std = df_weekly[feature_cols].std()
    
    normalized_data = (df_weekly[feature_cols] - data_mean) / data_std
    
    return df_weekly, normalized_data, feature_cols

# --- Environment ---
class RealMarketEnv:
    def __init__(self, raw_df, norm_df, feature_cols):
        self.raw_df = raw_df
        self.norm_df = norm_df
        self.feature_cols = feature_cols
        self.n_step = 0
        self.cash = 10000.0
        self.holdings = 0.0
        self.initial_balance = 10000.0
        self.history = []
        
    def reset(self):
        self.n_step = 0
        self.cash = 10000.0
        self.holdings = 0.0
        self.history = []
        return self._get_state()
        
    def _get_state(self):
        # Numeric State
        row = self.norm_df.iloc[self.n_step]
        numeric_state = row.values.astype(np.float32)
        
        # Text State (Narrative for DeepSeek)
        # We use the RAW values for the text, so the LLM understands "VIX is 30" (not "VIX is 2.5 sigma")
        raw_row = self.raw_df.iloc[self.n_step]
        
        # Simple template
        text_state = (
            f"Market Update: QQQ Return is {raw_row['QQQ_Log_Return']:.2%}. "
            f"VIX is {raw_row['VIX']:.1f}. RSI is {raw_row['RSI']:.1f}. "
            f"Yield is {raw_row['DGS2']:.2f}%. Inflation is {raw_row['CPI_YoY']:.1f}%. "
            f"Unemployment is {raw_row['Unemployment']:.1f}%."
        )
        
        # Portfolio State (Normalized roughly)
        port_state = np.array([
            self.cash / 10000.0, 
            self.holdings * raw_row['QQQ'] / 10000.0
        ], dtype=np.float32)
        
        return {
            "numeric": np.concatenate([numeric_state, port_state]), # Features + Portfolio
            "text": text_state,
            "price": raw_row['QQQ'],
            "date": self.raw_df.index[self.n_step]
        }
        
    def step(self, action):
        # 0=Hold, 1=Buy, 2=Sell
        current_price = self.raw_df.iloc[self.n_step]["QQQ"]
        
        # Execute Trade
        if action == 1: # Buy
            if self.cash > current_price:
                # Buy as much as possible (simplified) or just 1 unit?
                # Let's buy 1 unit for simplicity of learning
                cost = current_price
                if self.cash >= cost:
                    self.cash -= cost
                    self.holdings += 1
        elif action == 2: # Sell
            if self.holdings > 0:
                self.cash += current_price
                self.holdings -= 1
                
        # Move Step
        self.n_step += 1
        done = self.n_step >= len(self.raw_df) - 1
        
        # Calculate Reward
        # Reward = Change in Portfolio Value
        next_price = self.raw_df.iloc[self.n_step]["QQQ"] if not done else current_price
        new_value = self.cash + (self.holdings * next_price)
        prev_value = self.cash + (self.holdings * current_price) # Approx
        
        # Reward is the immediate profit/loss
        # We scale it down to keep gradients stable
        reward = (new_value - prev_value) / 10.0 
        
        return self._get_state(), reward, done, new_value

# --- Agent ---
class HybridAgent(nn.Module):
    def __init__(self, base_model, numeric_dim, hidden_size=2048):
        super().__init__()
        self.base_model = base_model
        # Freeze LLM
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Input: LLM Embedding (2048) + Numeric Features (7) + Portfolio (2)
        self.fc1 = nn.Linear(hidden_size + numeric_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3) # Q-Values
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask, numeric_input):
        # 1. LLM Path
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            text_emb = outputs.hidden_states[-1][:, -1, :] # [Batch, 2048]
            
        # 2. Combine
        x = torch.cat([text_emb.float(), numeric_input], dim=1)
        
        # 3. MLP Path
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- Main ---
if __name__ == "__main__":
    print("Loading Data...")
    df_weekly, norm_df, features = load_and_process_data(DATA_FILE)
    print(f"Loaded {len(df_weekly)} weekly records.")
    
    print("Loading DeepSeek...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16)
    
    # Numeric Dim = Features + 2 (Cash, Holdings)
    agent = HybridAgent(base_model, numeric_dim=len(features)+2)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    env = RealMarketEnv(df_weekly, norm_df, features)
    epsilon = EPSILON_START
    
    print("\n--- Starting Training (Weekly Data) ---")
    
    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        start_time = time.time()
        
        while not done:
            # Prepare Inputs
            text_tokens = tokenizer(state["text"], return_tensors="pt", truncation=True, max_length=128)
            numeric_tensor = torch.tensor([state["numeric"]], dtype=torch.float32)
            
            # Action
            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                with torch.no_grad():
                    q_values = agent(text_tokens.input_ids, text_tokens.attention_mask, numeric_tensor)
                    action = torch.argmax(q_values).item()
            
            # Step
            next_state, reward, done, port_value = env.step(action)
            
            # Train (Online / No Replay Buffer for simplicity of demo script)
            # Ideally use ReplayBuffer, but we want to see it run sequentially
            next_text = tokenizer(next_state["text"], return_tensors="pt", truncation=True, max_length=128)
            next_numeric = torch.tensor([next_state["numeric"]], dtype=torch.float32)
            
            with torch.no_grad():
                target_q = agent(next_text.input_ids, next_text.attention_mask, next_numeric)
                max_next_q = torch.max(target_q).item()
                target = reward + GAMMA * max_next_q
                
            current_q = agent(text_tokens.input_ids, text_tokens.attention_mask, numeric_tensor)
            target_vec = current_q.clone()
            target_vec[0][action] = target
            
            loss = loss_fn(current_q, target_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward
            
            if env.n_step % 100 == 0:
                print(f"  Week {env.n_step}: Value=${port_value:.0f} | Epsilon={epsilon:.2f}")
                
        # End Episode
        duration = time.time() - start_time
        print(f"Episode {episode+1} Finished. Total Reward: {total_reward:.2f}. Final Value: ${port_value:.2f}. Time: {duration:.1f}s")
        
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            
    print("\nTraining Complete.")
