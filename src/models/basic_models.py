import torch
import torch.nn as nn

from .model_utils import get_transformer

class FlatTransModel(nn.Module):
    def __init__(self, system, class_num=1):
        super().__init__()
        self.transformer = get_transformer(system)
        self.classifier = nn.Linear(768, class_num)

    def forward(self, x, mask):
        h1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state[:,0]
        y = self.classifier(h1)
        return y

class FlatSegModel(nn.Module):
    def __init__(self, system, class_num=1):
        super().__init__()
        self.transformer = get_transformer(system)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, mask):
        h1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state
        y = self.classifier(h1).squeeze(-1)
        y = self.sigmoid(y)
        return y
    
    
    
    