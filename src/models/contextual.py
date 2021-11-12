import torch
import torch.nn as nn
import math

from .basic_models import get_transformer

class SpanModel(nn.Module): 
    def __init__(self, system, num_classes):
        super().__init__()
        self.transformer = get_transformer(system)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x, segs):
        x = x.unsqueeze(0)
        h1 = self.transformer(input_ids=x).last_hidden_state.squeeze(0)
        utt_vecs = self.get_utt_vec(h1, segs)
        y = self.classifier(utt_vecs)
        return y

    def get_utt_vec(self, h1, segs):
        utt_vecs = torch.zeros(len(segs)-1, 768).to(h1.device)
        for k in range(0, len(segs)-1):
            utt_vector = (h1[k] + h1[k+1])/2
            utt_vecs[k] = utt_vector
        return utt_vecs

