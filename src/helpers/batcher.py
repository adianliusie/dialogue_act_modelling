import torch
from typing import List, Tuple
from types import SimpleNamespace
import random
from abc import abstractmethod, ABCMeta

from ..utils import flatten


class BaseBatcher(metaclass=ABCMeta):
    """base batching helper class to be inherited"""

    def __init__(self, max_len:int=None):
        self.max_utt_len = max_len
        self.device = torch.device('cpu')
    
    def to(self, device:torch.device):
        self.device = device

    def batches(self, data:'ConvHelper', bsz:int=8, shuf:bool=False):
        convs = self.prep_convs(data)
        utts = [utt for conv in convs for utt in conv]
        if shuf: random.shuffle(utts)
        batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
        batches = [self.batchify(batch) for batch in batches] 
        if self.max_utt_len:
            assert max([len(batch.ids[0]) for batch in batches])<= self.max_utt_len       
        return batches

    def conv_batches(self, data:'ConvHelper', bsz:int=None):
        """batches an entire conversation, and provides conv id"""
        conv_ids = [conv.conv_id for conv in data]
        convs = self.prep_convs(data)
        for conv_id, conv in zip(conv_ids, convs):
            conv = self.batchify(conv)
            yield (conv_id, conv)
        
    def get_padded_ids(self, ids:list):
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

    @abstractmethod
    def batchify(self):
        pass
    
    @abstractmethod
    def prep_convs(self):
        pass
    

class DActsBatcher(BaseBatcher):
    """batching helper for DA prediction"""
    
    def __init__(self, mode:str, mode_args:Tuple[int]=(0,0), max_len:int=None):
        super().__init__(max_len)
        self.past, self.fut = mode_args
   
    def batchify(self, batch):
        """each input is input ids and mask for utt, + label"""
        ids, labels = zip(*batch)  
        ids, mask = self.get_padded_ids(ids)
        labels = torch.LongTensor(labels).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels)

    def prep_convs(self, data:'ConvHelper'):    
        output = []
        for conv in data:
            conv_out = []
            for i, cur_utt in enumerate(conv.utts):
                past_utts = [utt.ids[1:-1] for utt in conv.utts[max(i-self.past, 0):i]]
                future_utts = [utt.ids[1:-1] for utt in conv.utts[i+1:i+self.fut+1]]
                ids = self.utt_join(past_utts, cur_utt.ids, future_utts)
                conv_out.append([ids, cur_utt.label])
            output.append(conv_out)
        return output

    def prep_conv_ids(self, conv_ids):
        output = []
        for k, curr_ids in enumerate(conv_ids):
            past_utts = [utt_ids[1:-1] for utt_ids in conv_ids[max(k-self.past, 0):k]]
            future_utts = [utt_ids[1:-1] for utt_ids in conv_ids[k+1:k+self.fut+1]]
            ids = self.utt_join(past_utts, curr_ids, future_utts)
            output.append(ids)
        return output
                
    def utt_join(self, past, cur, fut):
        if self.max_utt_len and max(len(past), len(fut)) != 0:
            k = 0
            while len(flatten(past)+cur+flatten(fut)) > self.max_utt_len and len(past)>0:
                if k%2 == 0: past = past[1:]
                else:        fut  = fut[:-1]
                k += 1
                
        output = flatten(past) + cur + flatten(fut)
        return output

    
class SegBatcher(BaseBatcher):
    """batching helper for phrase segmentation"""
    
    def __init__(self, max_len:int=None):
        super().__init__(max_len)
    
    def batchify(self, batch):
        ids, segs = zip(*batch)   
        ids, mask = self.get_padded_ids(ids)
        if segs: labels = self.one_hot_encode(segs, len(ids[0])).to(self.device)
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, segs=segs)

    def one_hot_encode(self, labels, max_len):
        """ creates label matrix for training"""
        
        output = torch.zeros(len(labels), max_len)
        for k, lab in enumerate(labels):
            for l in lab:
                output[k, l] = 1
        return output

    def prep_convs(self, data:'ConvHelper'):
        output = []
        for conv in data:
            conv_out = []
            for turn in conv.turns:
                ids  = turn.ids
                segs = turn.tags['segs'] if type(turn.tags)==dict and 'segs' in turn.tags else None
                conv_out.append([ids, segs])
            output.append(conv_out)
        return output
    
    
    
                 
                 
                 
                 
                 
                 
                 