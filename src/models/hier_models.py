import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel
from types import SimpleNamespace #TEMP

from .model_utils import Attention, get_transformer 

class HierModel(nn.Module):
    def __init__(self, system, class_num, decoder='baseline', layers=1):  
        super().__init__()
        self.utt_encoder = get_transformer(system)
        self.conv_encoder = Decoder(system=decoder, layers=layers)
        
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
        h = self.pooling(H1, mask).unsqueeze(0)
        h2 = self.conv_encoder(h)
        y = self.classifier(h2).squeeze(0)
        return y

class Decoder(nn.Module):
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

class AutoRegressive(nn.Module):
    def __init__(self, system, class_num):  
        super().__init__()
        self.class_num = class_num
        self.utt_encoder = get_transformer(system)
        self.embedding = nn.Embedding(class_num, 4)
        self.decoder = nn.LSTM(input_size=772, hidden_size=772, num_layers=1, bias=True, 
                               batch_first=True, dropout=0, bidirectional=False)
        self.pooling = lambda ids, mask: ids[:,0]
        self.classifier = nn.Linear(772, class_num)

    def forward(self, ids, mask, y):
        H1 = self.utt_encoder(ids, mask).last_hidden_state #encoder
        h = self.pooling(H1, mask).unsqueeze(0)
        y_inp = torch.roll(y, shifts=1, dims=0).unsqueeze(0)
        y_embed = self.embedding(y_inp)
        y_embed[0, 0, :] = 0
        h2 = torch.cat((h, y_embed), -1)
        h3 = self.decoder(h2)[0]
        y = self.classifier(h3).squeeze(0)
        return y

    def decode(self, ids, mask):
        H1 = self.utt_encoder(ids, mask).last_hidden_state #encoder
        H = self.pooling(H1, mask)
        
        y_embed = torch.zeros(4, device=H.device)
        hn = cn = torch.zeros(1, 1, 772, device=H.device)
        output = torch.zeros([len(H), self.class_num], device=H.device)
        for k, h in enumerate(H):
            inp = torch.cat((h, y_embed), -1).unsqueeze(0).unsqueeze(0)
            h_out, (hn,cn) = self.decoder(inp, (hn,cn))
            y = self.classifier(h_out.squeeze(0))
            pred = torch.argmax(y, dim=-1).squeeze(0)
            pred[pred != -100] = 1
            y_embed = self.embedding(pred)
            output[k,:] = y.clone()
        return output
