import random
from types import SimpleNamespace
import torch

class SegmentHandler:
    def __init__(self):
        self.device = torch.device('cpu')
       
    def batches(self, data, bsz=8):
        convs = self.prep_conv(data)
        utts = [utt for conv in convs for utt in conv]
        random.shuffle(utts)
        batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
        batches = [self.batchify(batch) for batch in batches] 
        return batches
    
    def seg_conv(self, data, bsz=8):
        convs = self.prep_conv(data)
        for conv in convs:
            batches = [conv[i:i+bsz] for i in range(0,len(conv), bsz)]
            batches = [self.batchify(batch) for batch in batches] 
            yield batches

    def batches_seg(self, ids, align, labels, bsz=8):
        for conv_ids, conv_align, conv_labels in zip(ids, align, labels):
            utts = list(zip(conv_ids, conv_align, conv_labels))
            batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
            batches = [self.batchify_seg(batch) for batch in batches]
            yield batches
            
    def batchify(self, batch):
        ids, segs, acts = zip(*batch)   
        ids, mask = self.get_padded_ids(ids)
        labels = self.one_hot_encode(segs, len(ids[0])).to(self.device)
        act_dict = [{i:act for i, act in zip(seg_ex, acts_ex)} for seg_ex, acts_ex in zip(segs, acts)]
        segs = [[1] + seg for seg in segs]
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, segs=segs, act_dict=act_dict)

    def batchify_seg(self, batch):
        ids, align, labels = zip(*batch)  
        ids, mask = self.get_padded_ids(ids)
        return SimpleNamespace(ids=ids, mask=mask, align=align, labels=labels)
    
    def get_padded_ids(self, ids):
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
    
    def one_hot_encode(self, segs, ids_len):
        output = torch.zeros(len(segs), ids_len)
        for k, seg in enumerate(segs):
            for s in seg:
                output[k, s] = 1
        return output

    def prep_conv(self, data):
        output = []
        for conv in data:
            conv_out = []
            for turn in conv.turns:
                ids  = turn.ids
                segs = turn.segs
                acts = turn.acts
                conv_out.append([ids, segs, acts])
            output.append(conv_out)
        return output

    def to(self, device):
        self.device = device
