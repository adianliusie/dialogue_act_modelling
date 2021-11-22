import torch
import torch.nn as nn

from .model_utils import get_transformer

class FlatTransModel(nn.Module):
    def __init__(self, system, class_number=1):
        super().__init__()
        self.transformer = get_transformer(system)
        self.classifier = nn.Linear(768, class_number)

    def forward(self, x, mask):
        h1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state[:,0]
        y = self.classifier(h1)
        return y
