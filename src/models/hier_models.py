import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel
from types import SimpleNamespace #TEMP

from .model_utils import Attention, get_transformer 

class HierModel(nn.Module):
    def __init__(self, system, class_num, context='baseline', layers=1):  
        super().__init__()
        self.utt_encoder = get_transformer(system)
        self.conv_encoder = ContextModel(system=context, layers=layers)
        self.pooling = self.get_pooling()           #self.get_pooling(pooling)
        self.classifier = nn.Linear(768, class_num)
        
    def get_pooling(self, pooling='first'):
        if pooling == 'attention':
            self.attention = Attention(768)
            pooling_fn = lambda ids, mask=None: self.attention(ids, mask)
        elif pooling == 'first':
            pooling_fn = lambda ids, mask: ids[:,0]
        return pooling_fn  
    
    def forward(self, x, mask):
        H1 = self.utt_encoder(x, mask).last_hidden_state #encoder
        H1 = self.pooling(H1, mask).unsqueeze(0)
        H2 = self.conv_encoder(H1)
        y = self.classifier(H2).squeeze(0)
        return y

class AutoRegressiveModel(nn.Module):
    def __init__(self, system, class_num, context='baseline', layers=1):  
        super().__init__()
        self.class_num = class_num
        self.utt_encoder = get_transformer(system)
        self.embedding = nn.Embedding(class_num, 4)
        self.conv_encoder = ContextModel(system=context, layers=layers)
        self.decoder = nn.LSTM(input_size=772, hidden_size=772, num_layers=1, bias=True, 
                               batch_first=True, dropout=0, bidirectional=False)
        self.pooling = lambda ids, mask: ids[:,0]
        self.classifier = nn.Linear(772, class_num)

    def forward(self, ids, mask, y):
        H1 = self.utt_encoder(ids, mask).last_hidden_state #encoder
        H1 = self.pooling(H1, mask).unsqueeze(0)

        y_inp = torch.roll(y, shifts=1, dims=0)
        y_embed = self.embedding(y_inp).unsqueeze(0)
        y_embed[0, 0, :] = 0
        
        H2 = self.conv_encoder(H1)
        H2 = torch.cat((H2, y_embed), -1)
        H3 = self.decoder(H2)[0]
        y = self.classifier(H3).squeeze(0)
        return y

    def decode(self, ids, mask):
        H1 = self.utt_encoder(ids, mask).last_hidden_state #encoder
        H1 = self.pooling(H1, mask).unsqueeze(0)
        H2 = self.conv_encoder(H1).squeeze(0)

        y_embed = torch.zeros(4, device=H2.device)
        hn = cn = torch.zeros(1, 1, 772, device=H2.device)
        output = torch.zeros([len(H2), self.class_num], device=H2.device)

        for k, h_k in enumerate(H2):
            x_k = torch.cat((h_k, y_embed), -1).unsqueeze(0).unsqueeze(0)
            h_out, (hn,cn) = self.decoder(x_k, (hn,cn))
            y = self.classifier(h_out.squeeze(0))
            pred = torch.argmax(y, dim=-1).squeeze(0)
            y_embed = self.embedding(pred)
            output[k,:] = y.clone()
        return output

class ContextModel(nn.Module):
    def __init__(self, system, layers):
        super().__init__()
        print(system)
        if system   == 'baseline'    : self.baseline()
        if system   == 'fcc'         : self.fcc()
        elif system == 'bilstm'      : self.bilstm(layers)
        elif system == 'transformer' : self.transformer(layers)
        elif system == 'attention'   : self.self_attention()
        elif system == 'ctx_atten'   : self.context_attention()

    def baseline(self):
        self.forward = lambda x: x

    def fcc(self):
        self.model = nn.Linear(768, 768)
        self.forward = lambda x: self.model(x)

    def bilstm(self, layers):
        self.model = nn.LSTM(input_size=768, hidden_size=768//2, num_layers=1, bias=True, 
                             batch_first=True, dropout=0, bidirectional=True)  
        self.forward = lambda x: self.model(x)[0]
 
    def transformer(self, layers):
        config = BertConfig(hidden_size=768, num_hidden_layers=layers, num_attention_heads=768//64, 
                            intermediate_size=4*768, return_dict=True)
        self.model = BertModel(config)
        self.forward = lambda x: self.model(inputs_embeds=x).last_hidden_state

    def self_attention(self):
        self.model = nn.MultiheadAttention(embed_dim=768, num_heads=1)
        self.forward = lambda x: self.model(x, x, x)[0]

    def context_attention(self):
        def context_mask(x, past=5, future=2):
            x_len = len(x)
            lower = torch.tril(torch.ones([x_len,x_len], device=x.device), diagonal=future)
            upper = torch.triu(torch.ones([x_len,x_len], device=x.device), diagonal=-past)
            return lower*upper

        self.model = nn.MultiheadAttention(embed_dim=768, num_heads=1)
        self.forward = lambda x: self.model(x, x, x, attn_mask=context_mask(x))[0]
