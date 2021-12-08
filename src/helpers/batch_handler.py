import torch
import random 
from types import SimpleNamespace
import numpy as np

class BatchHandler:
    def __init__(self, mode, mode_args, max_len_conv=None, max_len_utt=None):
        self.device = torch.device('cpu')
        self.mode = mode
        self.past, self.fut = mode_args
        self.max_c = max_len_conv
        self.max_u = max_len_utt

    def batches(self, data, bsz=8):
        if self.mode in ['independent', 'context']: 
            output = self.batches_flat(data, bsz)
        if self.mode in ['hier', 'auto_regressive']: 
            output = self.batches_hier(data)
        return output
    
    def batches_flat(self, data, bsz=8):
        convs = self.prep_conv(data)
        utts = [utt for conv in convs for utt in conv]
        random.shuffle(utts)
        batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
        batches = [self.batchify(batch) for batch in batches] 
        batches = [batch for batch in batches if len(batch.ids[0]) <= self.max_u]        
        return batches

    def batches_hier(self, data):
        convs = self.prep_conv(data)
        convs = [conv[i:i+self.max_c] for conv in convs for i in range(0,len(conv), self.max_c)]
        convs = [conv for conv in convs if len(conv) >= min(self.max_c, 50)]
        random.shuffle(convs)
        batches  = [self.batchify(conv) for conv in convs]   
        batches  = [batch for batch in batches if len(batch.ids[0]) <= self.max_u]
        return batches

    def batchify(self, batch):
        ids, labels = zip(*batch)    
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels)

    def batches_conv(self, batch):
        examples = self.prepare_fn(data)
        random.shuffle(examples)
        batches = [conv for conv in examples if len(conv[0])<self.max_len]
        batches = [self.batchify_flat(conv) for conv in examples]        

    def batchify_flat(self, batch):
        ids, labels, info = batch
        ids = torch.LongTensor(ids).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=info, labels=labels)

    def to(self, device):
        self.device = device

    def shorten_doc(self, conv):
        conv = conv.copy()
        if self.max_len != None:
            while len([word for sent in document for word in sent[1:-1]]) > self.max_len:
                document.pop(-1)
        return document

    def prep_conv(self, data):
        #reduces context utterance to max utterance length
        def utt_join(past, cur, fut):
            if self.max_u:
                while len(past + cur + fut) > self.max_u:
                    if len(past) == 0 or len(fut) == 0: 
                        past, fut = [], []
                        break
                    past, fut = past[1:], fut[:-1]
            return past + cur + fut
                
        if self.mode == 'independent': self.past = self.fut = 0
        output, context = [], []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                past_utts = [tok for utt in conv.utts[max(i-self.past, 0):i] for tok in utt.ids[1:-1]]
                future_utts = [tok for utt in conv.utts[i+1:i+self.fut+1] for tok in utt.ids[1:-1]]
                ids = utt_join(past_utts, cur_utt.ids, future_utts)
                conv_out.append([ids, cur_utt.act])
            output.append(conv_out)
        return output
