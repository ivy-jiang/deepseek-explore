import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from agent import HybridAgent

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
print("Loading model for dummy weights...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16)
agent = HybridAgent(base_model, numeric_dim=9) # 7 features + 2 portfolio
torch.save(agent.state_dict(), "hybrid_model.pth")
print("Saved dummy hybrid_model.pth")
