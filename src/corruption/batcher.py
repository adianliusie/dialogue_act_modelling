from collections import namedtuple
from tqdm import tqdm
import torch
import random 

from .corruption import create_corrupted_set
from .tokenizer import get_tokenizer

flatten = lambda doc: [word for sent in doc for word in sent]
Batch = namedtuple('Batch', ['ids', 'mask'])
ScoreBatch = namedtuple('ScoreBatch', ['ids', 'mask', 'score'])

class Batcher:
    def __init__(self, system, hier=False, bsz=8, max_len=512, schemes=[1], args=None):
        self.hier = hier 
        self.bsz = bsz
        self.max_len = max_len
        self.schemes = schemes
        self.args = args
        
        self.tokenizer = get_tokenizer(system)
        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id
        
        self.device = torch.device('cpu')
        
    def batches(self, documents, c_num):
        coherent = documents.copy()
        random.shuffle(coherent)
        coherent = [self.tokenize_doc(doc.sents) for doc in tqdm(coherent)]
        coherent = [self.shorten_doc(doc) for doc in coherent]
        
        cor_pairs = self.corupt_pairs(coherent, c_num)
        batches = [cor_pairs[i:i+self.bsz] for i in range(0,len(cor_pairs), self.bsz)]
        batches = [self.batch_pair(batch) for batch in batches]
        return batches

    def corupt_pairs(self, coherent, c_num):
        incoherent = [create_corrupted_set(doc, c_num, self.schemes, self.args) for doc in coherent]
        examples = []
        for pos, neg_set in zip(coherent, incoherent):
            for neg in neg_set:
                examples.append([pos, neg])
        return examples
    
    def batch_pair(self, doc_pairs):
        if self.hier == False:
            coherent, incoherent = zip(*doc_pairs)
            coherent = [self.flatten_doc(doc) for doc in coherent]
            incoherent = [self.flatten_doc(doc) for doc in incoherent]
            pos_batch = self.batchify(coherent)
            neg_batch = self.batchify(incoherent)
            batch = pos_batch, neg_batch
        else:                
            batch = [[self.batchify(coh), self.batchify(inc)] for coh, inc in doc_pairs]
        return batch
    
    def labelled_batches(self, documents, shuffle=True):
        documents = documents.copy()
        if shuffle: random.shuffle(documents)
        sents = [self.tokenize_doc(doc.sents) for doc in documents]
        scores = [doc.score for doc in documents]
        batches = [(sents[i:i+self.bsz], scores[i:i+self.bsz]) for i in range(0,len(sents), self.bsz)]
        batches = [self.batch_lab(*batch) for batch in batches]
        return batches
    
    def batch_lab(self, documents, scores):
        if self.hier == False:
            documents = [self.flatten_doc(doc) for doc in documents]
            batch = self.batchify_s(documents, scores)
        if self.hier == True:
            batch = [self.batchify_s(doc, [score]) for doc, score in zip(documents, scores)]
        return batch     
        
    def tokenize_doc(self, document):
        return [self.tokenizer(sent).input_ids for sent in document]

    def flatten_doc(self, document):
        if self.CLS != None: 
            document = [sent[1:-1] for sent in document]
            ids = [self.CLS] + flatten(document) + [self.SEP]
        else:
            ids = flatten(document)
        return ids
        
    def filter_docs(self, documents, max_len=None):
        if max_len is None: max_len = self.max_len
        if not self.hier: documents = [doc for doc in documents if len(self.flatten_doc(doc)) < max_len] 
        else:  documents = [doc for doc in documents if max([len(i) for i in doc]) < max_len] 
        return documents
    
    def shorten_doc(self, document):
        document = document.copy()
        if self.hier:
            document = [sent[:self.max_len-1] for sent in document]       
        elif not self.hier:
            while len(self.flatten_doc(document)) > self.max_len:
                document.pop(-1)
        return document
    
    def batchify_s(self, batch, scores):
        batch = self.batchify(batch)
        scores = torch.FloatTensor(scores).to(self.device)
        return ScoreBatch(batch.ids, batch.mask, scores)
    
    def batchify(self, batch):
        max_len = max([len(i) for i in batch])
        ids = [doc + [0]*(max_len-len(doc)) for doc in batch]
        mask = [[1]*len(doc) + [0]*(max_len-len(doc)) for doc in batch]
        ids = torch.LongTensor(ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return Batch(ids, mask)

    def to(self, device):
        self.device = device
