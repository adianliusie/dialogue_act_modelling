import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve

from .helpers import ConvHandler, BatchHandler, SegmentHandler, Logger
from .models import FlatTransModel, SpanModel, HierModel, AutoRegressiveModel, FlatSegModel
from .utils import no_grad, toggle_grad, LossFunctions

class ExperimentHandler:
    def __init__(self, system_cfg):
        self.L = Logger(system_cfg)
        if system_cfg.load: system_cfg = self.L.system_cfg
        self.D = ConvHandler(system_cfg.data_src, system_cfg.system, system_cfg.punct, system_cfg.action, system_cfg.debug, 
                             system_cfg.class_reduct)
        self.B = BatchHandler(system_cfg.mode, system_cfg.mode_arg, system_cfg.max_len_conv, system_cfg.max_len_utts)

        self.mode = system_cfg.mode
        num_classes = max(self.D.act_id_dict.values())+1
        if self.mode in ['independent', 'context']: self.model = FlatTransModel(system_cfg.system, num_classes)
        elif self.mode == 'hier': self.model = HierModel(system_cfg.system, num_classes, system_cfg.hier_model, system_cfg.layers)
        elif self.mode == 'auto_regressive':
            self.model = AutoRegressiveModel(system_cfg.system, num_classes, system_cfg.hier_model, system_cfg.layers)
        elif self.mode == 'full_context': self.model = SpanModel(system_cfg.system, num_classes)
            
        self.device = torch.device(system_cfg.device) if torch.cuda.is_available() \
                      else torch.device('cpu')
        
        self.S = SegmentHandler()
        self.seg_model = FlatSegModel(self.L.system_cfg.system)

        self.cross_loss = torch.nn.CrossEntropyLoss()
        self.BCE_loss = torch.nn.BCELoss(reduction='none')
        
    def model_output(self, batch, train=True):
        if self.mode in ['independent', 'context', 'hier']: 
            y = self.model(batch.ids, batch.mask)
        elif self.mode in ['full_context']: 
            y = self.model(batch.ids, batch.info)
        elif self.mode in ['auto_regressive']: 
            if train: y = self.model(batch.ids, batch.mask, batch.labels)
            else:     y = self.model.decode(batch.ids, batch.mask)
        return y
    
    def train(self, config):
        self.to(self.device)
        self.L.save_config('train_cfg', config)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        if config.scheduling:
              SGD_steps = (len(train_data)*self.epochs)/config.bsz
              lambda1 = lambda i: 10*i/SGD_steps if i <= SGD_steps/10 else 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
              scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

        best_metric = 0
        for epoch in range(config.epochs):
            logger = np.zeros(3)

            for k, batch in enumerate(self.B.batches(self.D.train, config.bsz), start=1):
                y = self.model_output(batch)
                loss = self.cross_loss(y, batch.labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()
                
                y_pred = torch.argmax(y, -1)

                logger[0] += loss.item()
                logger[1] += sum(y_pred == batch.labels)
                logger[2] += len(batch.labels)

                if k%config.print_len == 0:
                    self.L.log(f'{k:<5}  {logger[0]/config.print_len:.3f}    {logger[1]/logger[2]:.3f}')
                    logger = np.zeros(3)
                
            preds, labels = self.evaluate(mode='dev')
            decision = np.argmax(preds, axis=-1)
            acc = sum(decision==labels)/len(decision)
            
            if acc > best_metric:
                self.save_model('best_epoch')
                best_metric = acc
            self.L.log(acc)
            
        self.load_model('best_epoch')
    
    @no_grad
    def evaluate(self, mode='dev'):
        self.to(self.device)
        if   mode ==  'dev': dataset = self.D.dev 
        elif mode == 'test': dataset = self.D.test
        predicted_probs, labels = [], []
        for k, batch in enumerate(self.B.batches(dataset), start=1):
            y = self.model_output(batch, train=False)
            loss = self.cross_loss(y, batch.labels)
            pred_prob = F.softmax(y, dim=-1)
            predicted_probs += pred_prob.cpu().tolist()
            labels += batch.labels.cpu().tolist()
        return(predicted_probs, labels)

    def train_seg(self, config=None):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=1e-5)

        logger = np.zeros(6)
        for k, batch in enumerate(self.S.batches(self.D.train), start=1):
            y = self.seg_model(batch.ids, batch.mask)
            loss = torch.mean(self.BCE_loss(y, batch.labels)[batch.mask==1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            mask = batch.mask
            logger[0] += ((y>0.5)==batch.labels)[mask==1].sum()
            logger[1] += mask.sum()
            logger[2] += (y>0.5)[mask*batch.labels==1].sum()
            logger[3] += (y>0.5).sum()
            logger[4] += (mask*batch.labels).sum()
            logger[5] += loss.item()

            if k%100 == 0:
                self.L.log(f'{k:<5}  loss:{logger[5]/100:.3f}  acc:{logger[0]/logger[1]:.3f}  '  
                           f'p:{logger[2]/logger[3]:.3f}  r:{logger[2]/logger[4]:.3f}')
                logger = np.zeros(6)

    @no_grad
    def eval_seg(self):
        self.to(self.device)
        if   mode ==  'dev': dataset = self.D.dev 
        elif mode == 'test': dataset = self.D.test        
        for k, batch in enumerate(self.S.batches(dataset), start=1):
            y = seg_model(batch.ids, batch.mask)
            loss = torch.mean(self.BCE_loss(y, batch.labels)[batch.mask==1])
            
            if k%100 == 0:
                self.L.log(f'{k:<5}  acc:{logger[0]/logger[1]:.3f}   p:{logger[2]/logger[3]:.3f}   r:{logger[2]/logger[4]:.3f}')
                logger = np.zeros(5)

    @no_grad
    def classify_turns(self, mode='dev'):
        if   mode ==  'dev': dataset = self.D.dev 
        elif mode == 'test': dataset = self.D.test
            
        ids, align, acts = self.segment_convs(dataset)
        
        predicted_probs, labels = [], []
        for conv in self.S.batches_seg(ids, align, acts):
            for batch in conv:
                y = self.model(batch.ids, batch.mask)
                pred_prob = F.softmax(y, dim=-1)
                predicted_probs += pred_prob.cpu().tolist()
                labels += batch.labels
        return predicted_probs, labels
    
    @no_grad
    def segment_convs(self, data):
        pairs = lambda x: [(x[i], x[i+1]) for i in range(len(x)-1)]
        batch_pair = lambda x: [pairs(i) for i in x]
        
        out_ids, out_align, out_acts, utt_count = [], [], [], 0
        for conv_num, conv in enumerate(self.S.seg_conv(self.D.dev), start=1):
            utt_ids, utt_align, utt_acts = [], [], []
            for batch in conv:
                y = self.seg_model(batch.ids, batch.mask)

                # for each turn, determine utterance start/end position
                decisions = [[1] for _ in range(len(y))]
                for i, j in (y>0.5).nonzero(as_tuple=False):
                    if j < batch.mask[i].sum():
                        decisions[i].append(int(j))

                decisions = batch_pair(decisions)
                seg_labels = batch_pair(batch.segs)
                utt_count += sum([1 for turn in seg_labels for utt in turn])
                
                # split turns into utterances
                for k, turn in enumerate(batch.ids):
                    for start, end in decisions[k]:
                        segment = turn[start:end]
                        utt_ids.append(segment.tolist())
                        utt_align.append((start, end) in seg_labels[k])
                        if utt_align[-1]: utt_acts.append(batch.act_dict[k][end])
                        else:             utt_acts.append(-1)
                            
            out_ids.append(utt_ids), out_align.append(utt_align), out_acts.append(utt_acts)

        hits = sum([sum(x) for x in out_align])
        preds = sum([len(x) for x in out_align])
        self.L.log(f'SEG  P:{hits/preds:.3f}   R:{hits/utt_count:.3f}')
        return out_ids, out_align, out_acts
    
    def save_model(self, name):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), f'{self.L.path}/models/{name}.pt')
        self.model.to(device)

    def load_model(self, name):
        self.model.load_state_dict(torch.load(self.L.path + f'/models/{name}.pt'))

    def to(self, device):
        self.model.to(device)
        self.B.to(device)
        self.seg_model.to(device)
        self.S.to(device)
