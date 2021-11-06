import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel

class BasicBertModel(nn.Module):
    def __init__(self, class_number):
        super().__init__()
        self.transformer = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.classifier = nn.Linear(768, class_number)

    def forward(self, x, mask):
        h1 = self.transformer(input_ids=x, attention_mask=mask).last_hidden_state[:,0]
        y = self.classifier(h1)
        return y  