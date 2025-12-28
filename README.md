# Hybrid AI Trading System (DeepSeek + DQN)

This project implements a **Hybrid AI Trading Agent** that combines the semantic understanding of a Large Language Model (DeepSeek 1.3B) with the decision-making capabilities of Reinforcement Learning (DQN).

## 🧠 Model Architecture

The core of the system is the `HybridAgent`, which fuses two distinct data streams to make trading decisions (Buy, Hold, Sell).

### 1. Text Stream (The "Eyes")
-   **Input**: Market narratives generated from data (e.g., *"Inflation is 2.2%, Market is up..."*).
-   **Model**: **DeepSeek 1.3B** (`deepseek-coder-1.3b-instruct`).
-   **Process**: The text is fed into DeepSeek. We extract the **last hidden state** (a vector of 2048 numbers) which represents the model's semantic understanding of the market context.
-   **Note**: The DeepSeek model is **frozen** during training; we use it purely as a high-level feature extractor.

### 2. Numeric Stream (The "Charts")
-   **Input**: Raw financial and economic data.
-   **Features**:
    -   **QQQ Log Return**: Recent market performance.
    -   **VIX**: Market volatility index (Fear gauge).
    -   **RSI (Relative Strength Index)**: Momentum indicator.
    -   **MACD**: Trend-following momentum indicator.
    -   **2-Year Treasury Yield (DGS2)**: Interest rate environment.
    -   **CPI YoY**: Inflation rate.
    -   **Unemployment Rate**: Economic health.
    -   **Portfolio State**: Current Cash and Holdings.

### 3. Fusion & Decision (The "Brain")
-   **Fusion**: The **Text Embedding** (2048 dims) and **Numeric Features** (9 dims) are concatenated into a single vector.
-   **MLP (Multi-Layer Perceptron)**: This combined vector is passed through a trainable neural network:
    -   `Linear -> ReLU -> Linear -> ReLU -> Linear`
-   **Output**: Q-Values for 3 actions: **[HOLD, BUY, SELL]**. The action with the highest value is chosen.

---

## 🚀 Usage

### 1. Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate huggingface_hub pandas yfinance pandas_datareader
```

### 2. Fetch Data
Download 20+ years of market and economic data (Yahoo Finance + FRED):
```bash
python fetch_data.py
```
*Output: `market_data.csv`*

### 3. Train the Model
Train the DQN agent on the historical data. This will simulate trading over the dataset and update the MLP weights.
```bash
python train_hybrid.py
```
*Output: `hybrid_model.pth` (Saved Model Weights)*

### 4. Live Inference
Get a trading recommendation based on the latest available data point using the trained model:
```bash
python inference.py
```
*Output:*
```text
--- Recommendation ---
Action: HOLD
Reasoning: Model suggests staying on the sidelines.
```

---

## 📂 File Structure
-   `agent.py`: Defines the `HybridAgent` class (Neural Network).
-   `fetch_data.py`: Data collection pipeline.
-   `train_hybrid.py`: Training loop (RL simulation).
-   `inference.py`: Live prediction script.
-   `explore_model.py`: Utility to inspect DeepSeek architecture.

## License
[MIT](LICENSE)
