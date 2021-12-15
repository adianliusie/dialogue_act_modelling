import torch
import random 
from types import SimpleNamespace
import numpy as np

from ..utils import flatten

class BatchHandler:
    def __init__(self, mode, mode_args, conv_len=None, utt_len=None):
        self.mode = mode
        self.past, self.fut = mode_args
        self.max_c = conv_len
        self.max_u = utt_len
        self.device = torch.device('cpu')

    def act_batches(self, data, bsz=8):
        self.conv_prep = self.conv_prep_da
        self.batchify = self.batchify_da
        
        if self.mode in ['independent', 'context']: 
            batches = self.batches_flat(data, bsz)
        if self.mode in ['hier', 'auto_regressive']: 
            batches = self.batches_hier(data)
        return batches
    
    def seg_batches(self, data, bsz=8):
        self.conv_prep = self.conv_prep_seg
        self.batchify = self.batchify_seg
        output = self.batches_flat(data, bsz)
        return output
    
    def batches_flat(self, data, bsz=8, seg=False):
        convs = self.conv_prep(data)
        utts = [utt for conv in convs for utt in conv]
        random.shuffle(utts)
        batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
        batches = [self.batchify(batch) for batch in batches] 
        batches = [batch for batch in batches if len(batch.ids[0]) <= self.max_u]        
        return batches

    def batches_hier(self, data):
        convs = self.conv_prep(data)
        convs = [conv[i:i+self.max_c] for conv in convs for i in range(0,len(conv), self.max_c)]
        convs = [conv for conv in convs if len(conv) >= min(self.max_c, 50)]
        random.shuffle(convs)
        batches  = [self.batchify(conv) for conv in convs]   
        batches  = [batch for batch in batches if len(batch.ids[0]) <= self.max_u]
        return batches

    ############   Dialogue Act Classification Methods   ############
    def batchify_da(self, batch):
        ids, labels = zip(*batch)  
        ids, mask = self.get_padded_ids(ids)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels)

    def conv_prep_da(self, data):    
        if self.mode == 'independent': self.past = self.fut = 0
        output, context = [], []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                past_utts = [utt.ids[1:-1] for utt in conv.utts[max(i-self.past, 0):i]]
                future_utts = [utt.ids[1:-1] for utt in conv.utts[i+1:i+self.fut+1]]
                ids = self.utt_join(past_utts, cur_utt.ids, future_utts)
                conv_out.append([ids, cur_utt.label])
            output.append(conv_out)
        return output

    def utt_join(self, past, cur, fut):
        if self.max_u and max(len(past), len(fut)) != 0:
            k = 0
            while len(flatten(past)+cur+flatten(fut)) > self.max_u and len(past)>0:
                if k%2 == 0: past = past[1:]
                else:        fut  = fut[:-1]
                k += 1
        return flatten(past) + cur + flatten(fut)
        
    ################     Segmentation Methods    ################
    def batchify_seg(self, batch):
        ids, segs, acts = zip(*batch)   
        ids, mask = self.get_padded_ids(ids)
        if segs and acts:
            labels = self.one_hot_encode(segs, len(ids[0])).to(self.device)
            act_dict = [{i:act for i, act in zip(seg_ex, acts_ex)} for seg_ex, acts_ex in zip(segs, acts)]
            segs = [[1] + seg for seg in segs]
        else:
            labels, segs, act_dict = None, None, None
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, segs=segs, act_dict=act_dict)

    def conv_prep_seg(self, data):
        output = []
        for conv in data:
            conv_out = []
            for turn in conv.turns:
                ids  = turn.ids
                segs = turn.tags['segs']   if type(turn.tags)==dict and 'segs'   in turn.tags else None
                acts = turn.tags['labels'] if type(turn.tags)==dict and 'labels' in turn.tags else None
                conv_out.append([ids, segs, acts])
            output.append(conv_out)
        return output

    def one_hot_encode(self, labels, max_len):
        output = torch.zeros(len(labels), max_len)
        for k, lab in enumerate(labels):
            for l in lab:
                output[k, l] = 1
        return output

    ################       Cascade Methods       ################
    def cascade_seg(self, data, bsz=8):
        convs = self.conv_prep_seg(data)
        for conv in convs:
            batches = [conv[i:i+bsz] for i in range(0, len(conv), bsz)]
            batches = [self.batchify_seg(batch) for batch in batches] 
            yield batches
    
    def cascade_act(self, ids, align, labels, bsz=8):
        ids = self.conv_prep_casc(ids)
        for conv_ids, conv_align, conv_labels in zip(ids, align, labels):
            utts = list(zip(conv_ids, conv_align, conv_labels))
            batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
            batches = [self.batchify_casc(batch) for batch in batches]
            yield batches
    
    def conv_prep_casc(self, data):    
        if self.mode == 'independent': self.past = self.fut = 0
        output, context = [], []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv):
                past_utts = [utt[1:-1] for utt in conv[max(i-self.past, 0):i]]
                future_utts = [utt[1:-1] for utt in conv[i+1:i+self.fut+1]]
                ids = self.utt_join(past_utts, cur_utt, future_utts)
                conv_out.append(ids)
            output.append(conv_out)
        return output
    
    def structure_ids(self, convs):
        convs = [[SimpleNamespace(ids=utt_ids) for utt_ids in conv] for conv in convs]
        convs = [SimpleNamespace(utts=conv) for conv in convs]
        return convs
    
    def batchify_casc(self, batch):
        ids, align, labels = zip(*batch)  
        ids, mask = self.get_padded_ids(ids)
        return SimpleNamespace(ids=ids, mask=mask, align=align, labels=labels)
    
    
    ################       General Methods       ################
    def to(self, device):
        self.device = device

    def get_padded_ids(self, ids):
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
