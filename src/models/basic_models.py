import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel, BigBirdModel, ReformerModel

class FlatTransModel(nn.Module):
    def __init__(self, system, class_number=1):
        super().__init__()
        self.transformer = get_transformer(system)
        self.classifier = nn.Linear(768, class_number)

    def forward(self, x, mask):
        h1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state[:,0]
        y = self.classifier(h1)
        return y

    
def get_transformer(name):
    if   name ==       'bert': transformer = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif name == 'bert_cased': transformer = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    elif name ==    'roberta': transformer = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif name ==    'electra': transformer = ElectraModel.from_pretrained('google/electra-base-discriminator', return_dict=True)
    elif name ==   'big_bird': transformer = BigBirdModel.from_pretrained('google/bigbird-roberta-base', return_dict=True)
    elif name == 'reformer': transformer = ReformerModel.from_pretrained('google/reformer-crime-and-punishment', return_dict=True)
    elif name ==       'rand': transformer = BertModel(BertConfig(return_dict=True))
    else: raise Exception
    return transformer