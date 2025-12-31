import torch
import pandas as pd
import numpy as np
import csv
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent import HybridAgent
from train_hybrid import load_and_process_data # Reuse data processing

# --- Configuration ---
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
DATA_FILE = "market_data.csv"
MODEL_PATH = "hybrid_model.pth"

def get_latest_recommendation():
    print("Loading Data...")
    # In a real live scenario, you would fetch the absolute latest data here
    # For now, we use the latest data point from our CSV
    df_weekly, norm_df, features = load_and_process_data(DATA_FILE)
    
    # Get the last row (most recent week)
    last_idx = -1
    raw_row = df_weekly.iloc[last_idx]
    norm_row = norm_df.iloc[last_idx]
    
    print(f"Analyzing data for week ending: {df_weekly.index[last_idx].date()}")
    
    # Prepare State
    # 1. Text State
    text_state = (
        f"Market Update: QQQ Return is {raw_row['QQQ_Log_Return']:.2%}. "
        f"VIX is {raw_row['VIX']:.1f}. RSI is {raw_row['RSI']:.1f}. "
        f"Yield is {raw_row['DGS2']:.2f}%. Inflation is {raw_row['CPI_YoY']:.1f}%. "
        f"Unemployment is {raw_row['Unemployment']:.1f}%."
    )
    print(f"Market Context: {text_state}")
    
    # 2. Numeric State (Features + Portfolio)
    # For recommendation, we assume a neutral portfolio state (Cash=10k, Holdings=0)
    # to see what the agent *would* do if starting fresh.
    # Or you can input your actual portfolio state.
    cash = 10000.0
    holdings = 0.0
    port_state = np.array([cash / 10000.0, holdings * raw_row['QQQ'] / 10000.0], dtype=np.float32)
    
    numeric_state = np.concatenate([norm_row.values.astype(np.float32), port_state])
    
    # --- Load Model ---
    print("Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16)
    
    agent = HybridAgent(base_model, numeric_dim=len(features)+2)
    
    # Load Weights
    try:
        agent.load_state_dict(torch.load(MODEL_PATH))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: Model weights not found. Run train_hybrid.py first.")
        return

    agent.eval()
    
    # --- Inference (Decision) ---
    text_tokens = tokenizer(text_state, return_tensors="pt", truncation=True, max_length=128)
    numeric_tensor = torch.tensor([numeric_state], dtype=torch.float32)
    
    with torch.no_grad():
        q_values = agent(text_tokens.input_ids, text_tokens.attention_mask, numeric_tensor)
        action = torch.argmax(q_values).item()

    # --- Generation (Explanation) ---
    # Ask DeepSeek to explain the situation verbally
    prompt = f"{text_state}\nBased on this data, provide a brief market sentiment analysis."
    gen_inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        gen_outputs = base_model.generate(**gen_inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
    
    llm_explanation = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    # Extract just the new part
    llm_explanation = llm_explanation.replace(prompt, "").strip()

    actions = ["HOLD", "BUY", "SELL"]
    action_str = actions[action]
    
    print("\n--- Recommendation ---")
    print(f"Action: {action_str}")
    print(f"LLM Insight: {llm_explanation}")
    print(f"Q-Values: {q_values.numpy()}")
    
    # --- Log to Tracker (CSV) ---
    csv_file = "trade_tracker.csv"
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Action", "Price", "LLM_Insight", "Q_Values"])
        
        writer.writerow([
            df_weekly.index[last_idx].date(),
            action_str,
            f"{raw_row['QQQ']:.2f}",
            llm_explanation,
            q_values.detach().numpy().tolist()
        ])
    print(f"Logged to {csv_file}")

if __name__ == "__main__":
    get_latest_recommendation()
