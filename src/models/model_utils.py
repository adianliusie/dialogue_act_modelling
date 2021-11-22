from transformers import BertConfig, BertModel, RobertaModel, ElectraModel, BigBirdModel, AlbertModel
import torch
import torch.nn as nn

def get_transformer(name):
    if   name ==       'bert': transformer = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif name == 'bert_cased': transformer = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    elif name ==    'roberta': transformer = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif name ==    'electra': transformer = ElectraModel.from_pretrained('google/electra-base-discriminator', return_dict=True)
    elif name ==   'big_bird': transformer = BigBirdModel.from_pretrained('google/bigbird-roberta-base', return_dict=True)
    elif name ==     'albert': transformer = AlbertModel.from_pretrained('albert-base-v2', return_dict=True)
    elif name ==       'rand': transformer = BertModel(BertConfig(return_dict=True))
    else: raise Exception
    return transformer

class Attention(nn.Module):
    tanh = nn.Tanh()
    softmax = nn.Softmax(dim=1)
    def __init__(self, dim=300):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1)
        
    def forward(self, x, mask=None):
        h1 = self.W(x)
        h1 = self.tanh(h1)
        s = self.v(h1)

        if torch.is_tensor(mask):
            s.squeeze(-1)[mask==0] = -1e5
        a = self.softmax(s)
        output = torch.sum(a*x, dim=1)
        return output
