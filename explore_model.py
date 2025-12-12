import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ID - using the Coder Instruct version as it's a robust 1.3B model
# You can also try 'deepseek-ai/deepseek-llm-7b-base' if you have more RAM, 
# but for 1.3B specifically, the coder models are very common on HF.
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

print(f"Loading model: {model_id}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("\n--- Model Architecture ---")
print(model)

print("\n--- Model Parameters ---")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

print("\n--- Layer Inspection ---")
# Inspect the first layer's attention weights if available
for name, param in model.named_parameters():
    if 'layers.0.self_attn' in name:
        print(f"Parameter: {name} | Shape: {param.shape} | Mean: {param.mean().item():.6f} | Std: {param.std().item():.6f}")

print("\n--- Simple Inference Test ---")
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(f"Input: {input_text}")
print(f"Output:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
