import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the pre-trained DeepSeek model
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"
print(f"Loading {model_id} for finance integration...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Load model and ensure it outputs hidden states
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    output_hidden_states=True
)

# 2. Define a custom PyTorch Trading Module
# This module uses the LLM to get text embeddings and then predicts a trading signal
class DeepSeekTradingSignal(nn.Module):
    def __init__(self, base_model, hidden_size=2048, num_classes=3):
        super().__init__()
        self.base_model = base_model
        # Freeze the base model to save memory/compute (optional)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # A simple classification head: Buy (0), Hold (1), Sell (2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get the output from DeepSeek
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the last hidden state of the last token as the sentence representation
        # hidden_states is a tuple, -1 is the last layer
        last_hidden_state = outputs.hidden_states[-1] 
        
        # Take the embedding of the last token (eos or last word)
        # Shape: [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        sentence_embedding = last_hidden_state[:, -1, :]
        
        # Pass through our custom trading head
        logits = self.classifier(sentence_embedding.float()) # Cast to float32 for stability
        return logits

# 3. Simulate a Finance Use Case
print("\n--- Initializing Custom Trading Model ---")
trading_model = DeepSeekTradingSignal(base_model)
trading_model.eval() # Set to eval mode

# Sample financial headlines
headlines = [
    "Company X reports record breaking Q3 earnings, stock expected to soar.",
    "Market uncertainty rises as inflation data misses expectations.",
    "CEO steps down amidst scandal, shares plummet in after-hours trading."
]

print("\n--- Processing Headlines ---")
for headline in headlines:
    inputs = tokenizer(headline, return_tensors="pt")
    
    # Forward pass through our custom model
    logits = trading_model(inputs.input_ids, inputs.attention_mask)
    probs = torch.softmax(logits, dim=-1)
    
    # Interpret result (random weights, so meaningless, but shows the mechanics)
    signal_idx = torch.argmax(probs).item()
    signals = ["BUY", "HOLD", "SELL"]
    
    print(f"Headline: {headline[:50]}...")
    print(f"  -> Raw Logits: {logits.detach().numpy()}")
    print(f"  -> Signal: {signals[signal_idx]} (Conf: {probs[0][signal_idx]:.2f})")

print("\nSuccess! You have integrated DeepSeek 1.3B with a custom PyTorch module.")
print("You can now train 'trading_model.classifier' on your own labeled financial data.")
