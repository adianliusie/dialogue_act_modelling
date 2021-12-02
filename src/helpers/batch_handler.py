import torch
import random 
from types import SimpleNamespace
import numpy as np

class BatchHandler:
    def __init__(self, mode, mode_args, max_len=None):
        self.device = torch.device('cpu')
        self.mode = mode
        self.prepare_fn = prep_fn(mode, mode_args)
        self.max_len = max_len
    
    def batches(self, data, bsz=8):
        if self.mode in ['independent', 'back_history', 'context']: 
            output = self.batches_indep(data, bsz)
        if self.mode in ['hier', 'auto_regressive']: 
            output = self.batches_hier(data)
        return output
    
    def batches_indep(self, data, bsz=8):
        examples = self.prepare_fn(data)
        random.shuffle(examples)
        examples = [conv[:100] for conv in examples] #TEMP
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        batches = [self.batchify(batch) for batch in batches] 
        batches = [batch for batch in batches if len(batch.ids[0]) <= self.max_len]        
        return batches

    def batches_hier(self, data):
        examples = utt_hier_fn(data)
        random.shuffle(examples)
        examples = [conv[:self.max_len] for conv in examples]
        batches  = [self.batchify(conv) for conv in examples]        
        batches  = [batch for batch in batches if len(batch.ids[0]) <= self.max_len]        
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

def prep_fn(mode, arg):
    if mode == 'independent' : func = independent_fn
    if mode == 'back_history': func = back_history(arg)
    if mode == 'context'     : func = context(*arg)
    if mode in ['hier', 'auto_regressive']: func = utt_hier_fn
    return func

def independent_fn(data):
    output = []
    for conv in data:
        for utt in conv:
            output.append([utt.ids, utt.act])
    return output

def back_history(context_len=5):
    def back_history_fn(data):
        output, context = [], []
        for conv in data:
            for utt in conv:
                context_ids = [i for u in context for i in u[1:-1]]
                ids = context_ids + utt.ids 
                output.append([ids, utt.act])
                context.append(utt.ids)
                context = context[-context_len:].copy()
        return output
    return back_history_fn

def context(past=3, future=3):
    def context_fn(data):
        output, context = [], []
        for conv in data:
            for i, cur_utt in enumerate(conv.utts):
                past_utts = [tok for utt in conv.utts[max(i-past, 0):i] for tok in utt.ids[1:-1]]
                future_utts = [tok for utt in conv.utts[i+1:i+future+1] for tok in utt.ids[1:-1]]
                ids = past_utts + cur_utt.ids + future_utts
                output.append([ids, cur_utt.act])
        return output
    return context_fn

def utt_hier_fn(data):
    output = []
    for conv in data:
        output.append([[utt.ids, utt.act] for utt in conv.utts])
    return output

def utt_flat():
    utt_hier_fn = utt_hier()
    def utt_flat_fn(data):
        data = utt_hier_fn(data)
        output = []
        for utts, labels in data:
            CLS, SEP = utts[0][0], utts[0][-1]
            flat_conv = [CLS] + [tok for utt in utts for tok in utt[1:-1]] + [SEP]
            span = [len(utt[1:-1]) for utt in utts]
            span_segments = [1] + list(1 + np.cumsum(span))
            output.append([flat_conv, labels, span_segments])
        return output
    return utt_flat_fn
