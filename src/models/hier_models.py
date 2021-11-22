import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel

from ..helpers.tokenizer import get_embeddings
from .model_utils import Attention, get_transformer 

class HierModel(nn.Module):
    def __init__(self, system, pooling='first'):  
        super().__init__()
        self.first_encoder = TransEncoder(system) 

        if pooling == 'attention':
            self.attention = Attention(768)
            self.pooling = lambda ids, mask=None: self.attention(ids, mask)
        elif pooling == 'first':
            self.pooling = lambda ids, mask: ids[:,0]
        
        if True: self.second_encoder = HierTransEncoder(768) 
        else:    self.second_encoder = HierBilstmEncoder(768) 

        if True:
            self.attention_2 = Attention(768)
            self.pooling_2 = lambda ids: self.attention(ids)
        else:
            self.pooling_2 = lambda ids: ids[:,0]

        self.classifier = nn.Linear(768, 1)

    def forward(self, x, mask):
        H1 = self.first_encoder(x, mask)
        h = self.pooling(H1, mask)
        H2 = self.second_encoder(h)
        h2 = self.pooling_2(H2)
        y = self.classifier(h2)
        return y

class TransEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.transformer = get_transformer(name)

    def forward(self, x, mask):
        H1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state
        return H1

class HierTransEncoder(nn.Module):
    def __init__(self, hsz=768):
        super().__init__()
        heads = hsz//64
        config = BertConfig(hidden_size=hsz, num_hidden_layers=2, num_attention_heads=heads, 
                            intermediate_size=4*hsz, return_dict=True)
        self.transformer = BertModel(config)

    def forward(self, x):
        H1 = self.transformer(inputs_embeds=x).last_hidden_state
        return H1 

class HierBilstmEncoder(nn.Module):
    def __init__(self, hsz=768):
        super().__init__()
        self.bilstm = nn.LSTM(input_size=hsz, hidden_size=hsz//2, num_layers=2, bias=True, 
                              batch_first=True, dropout=0, bidirectional=True)
        
    def forward(self, x):
        H1, _ = self.bilstm(x)
        return H1




