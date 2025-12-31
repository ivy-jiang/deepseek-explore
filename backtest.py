import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent import HybridAgent
from train_hybrid import load_and_process_data, RealMarketEnv

# --- Configuration ---
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
DATA_FILE = "market_data.csv"
MODEL_PATH = "hybrid_model.pth"

def run_backtest():
    print("Loading Data...")
    df_weekly, norm_df, features = load_and_process_data(DATA_FILE)
    
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16)
    
    agent = HybridAgent(base_model, numeric_dim=len(features)+2)
    try:
        agent.load_state_dict(torch.load(MODEL_PATH))
        print("Model weights loaded.")
    except FileNotFoundError:
        print("Model weights not found!")
        return

    agent.eval()
    env = RealMarketEnv(df_weekly, norm_df, features)
    
    # Metrics
    state = env.reset()
    done = False
    
    action_counts = {0: 0, 1: 0, 2: 0} # Hold, Buy, Sell
    trades = [] # (Type, Price, Profit) - Simplified tracking
    
    # We track "Virtual Trades" to calculate Hit Ratio
    # A "Trade" is defined as: Buy -> Sell (Round Trip)
    entry_price = 0
    in_position = False
    
    print("\n--- Running Backtest (Weekly Horizon) ---")
    
    while not done:
        # Prepare Input
        text_tokens = tokenizer(state["text"], return_tensors="pt", truncation=True, max_length=128)
        numeric_tensor = torch.tensor([state["numeric"]], dtype=torch.float32)
        
        # Action (Pure Exploitation)
        with torch.no_grad():
            q_values = agent(text_tokens.input_ids, text_tokens.attention_mask, numeric_tensor)
            action = torch.argmax(q_values).item()
            
        action_counts[action] += 1
        current_price = state["price"]
        
        # Track Trade Performance
        if action == 1: # Buy
            if not in_position and env.cash > current_price:
                entry_price = current_price
                in_position = True
        elif action == 2: # Sell
            if in_position:
                profit = current_price - entry_price
                trades.append(profit)
                in_position = False
                
        # Step
        next_state, reward, done, port_value = env.step(action)
        state = next_state
        
    # --- Report ---
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t > 0])
    hit_ratio = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print("\n" + "="*30)
    print("      BACKTEST RESULTS      ")
    print("="*30)
    print(f"Trading Horizon: Weekly (Friday Close)")
    print(f"Total Weeks: {len(df_weekly)}")
    print(f"Final Portfolio Value: ${port_value:,.2f}")
    print(f"Total Return: {((port_value - 10000)/10000)*100:.2f}%")
    print("-" * 20)
    print(f"Action Distribution:")
    print(f"  Hold: {action_counts[0]}")
    print(f"  Buy:  {action_counts[1]}")
    print(f"  Sell: {action_counts[2]}")
    print("-" * 20)
    print(f"Round-Trip Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Hit Ratio: {hit_ratio:.2f}%")
    print("="*30)

if __name__ == "__main__":
    run_backtest()
