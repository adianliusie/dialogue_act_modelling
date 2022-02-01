import torch
import torch.nn as nn

from ..utils import get_transformer, pairs

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
    
    def get_phrases(self, ids, mask, thresh=0.5):
        #convert all turns to phrase boundary predictions
        y = self.forward(ids, mask)
        
        phrases = []
        for turn_ids, probs in zip(ids, y):
            #for each turn, select all positions above threshold
            decisions = [1] + list((probs > thresh).nonzero(as_tuple=True)[0])
            segments = pairs(decisions)
            
            #Create phrases
            for start, end in segments:
                segment = turn_ids[start:end]
                phrases.append([101] + segment.tolist() + [102])

        return phrases
                                
    
    
    
    