import torch
import random 
from types import SimpleNamespace

class BatchHandler:
    def __init__(self, context, context_len):
        self.device = torch.device('cpu')
        self.prepare_fn = self.prepare_fn(context, context_len)
    
    def prepare_fn(self, context, context_len):
        if context == 'independent':
            func = self.independent_utts
        elif context == 'back_history':
            func = self.back_history(context_len)
        return func
    
    def independent_utts(self, data):
        output = []
        for conv in data:
            for utt in conv:
                output.append([utt.ids, utt.act])
        return output
    
    def back_history(self, context_len=5):
        def prepare_context(context):
            flat_context = [i for utt in context for i in utt[1:-1]]
            return flat_context

        def back_history_fn(data):
            output, context = [], []
            for conv in data:
                for utt in conv:
                    context_ids = prepare_context(context)
                    ids = context_ids + utt.ids 
                    output.append([ids, utt.act])
                    context.append(utt.ids)
                    context = context[-context_len:].copy()
            return output
        
        return back_history_fn
    

    def batches(self, data, bsz=8):
        examples = self.prepare_fn(data)
        random.shuffle(examples)
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        batches = [self.batchify(batch) for batch in batches]
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

    def to(self, device):
        self.device = device