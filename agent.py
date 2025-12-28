import torch
import torch.nn as nn

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
