import torch
import random 
from types import SimpleNamespace
import numpy as np

class BatchHandler:
    def __init__(self, mode, mode_args, max_len=None):
        self.device = torch.device('cpu')
        self.mode = mode
        self.prepare_fn = prep_ids(mode, mode_args)
        self.max_len = max_len
    
    def batches(self, data, bsz=8):
        examples = self.prepare_fn(data)
        random.shuffle(examples)

        if self.mode == 'full_context':
            batches = [conv for conv in examples if len(conv[0])<self.max_len]
            batches = [self.batchify_flat(conv) for conv in examples]        
        else:
            batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
            batches = [self.batchify(batch) for batch in batches]          
        return batches
    
    def batchify_flat(self, batch):
        ids, labels, info = batch
        ids = torch.LongTensor(ids).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=info, labels=labels)

    def batchify(self, batch):
        ids, labels = zip(*batch)
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels)

    def to(self, device):
        self.device = device

    def shorten_doc(self, document):
        document = document.copy()
        if self.max_len != None:
            while len([word for sent in document for word in sent[1:-1]]) > self.max_len:
                document.pop(-1)
        return document
    
def prep_ids(mode, arg):
    if mode == 'independent' : func = independent()
    if mode == 'back_history': func = back_history(arg)
    if mode == 'full_context': func = utt_flat()
    if mode == 'hier'        : func = utt_hier()
    return func

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

def utt_hier():
    def utt_hier_fn(data):
        output = []
        for conv in data:
            if len(conv.turns) > 1:
                turns = [utt.ids for utt in conv.utts]
                labels = [utt.act for utt in conv.utts]
                output.append([turns, labels])
        return output
    return utt_hier_fn
        
def independent():
    def independent_fn(data):
        output = []
        for conv in data:
            for utt in conv:
                output.append([utt.ids, utt.act])
        return output
    return independent_fn

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
