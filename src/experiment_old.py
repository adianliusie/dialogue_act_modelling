import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve
from types import SimpleNamespace
import matplotlib.pyplot as plt

from .helpers import ConvHandler, BatchHandler, Logger
from .models import FlatTransModel, SpanModel, HierModel, AutoRegressiveModel, FlatSegModel
from .utils import no_grad, toggle_grad, LossFunctions

class ExperimentHandler:
    def __init__(self, system_cfg):
        self.device = torch.device(system_cfg.device) if torch.cuda.is_available() else torch.device('cpu')

        load = system_cfg.load

        self.L = Logger(system_cfg)
        if load: system_cfg = self.L.system_cfg

        self.mode = system_cfg.mode
        self.system = system_cfg.system
        
        self.act_model = self.create_act_model(system_cfg, 43)
        self.seg_model = FlatSegModel(self.system)
        if load: self.load_models()
        self.B = BatchHandler(system_cfg.mode, system_cfg.mode_arg, system_cfg.max_len_conv, system_cfg.max_len_utts)

        self.cross_loss = torch.nn.CrossEntropyLoss()
        self.BCE_loss = torch.nn.BCELoss(reduction='none')
        
    def create_act_model(self, config, num_classes):
        if config.mode in ['independent', 'context']: 
            model = FlatTransModel(config.system, num_classes)
        elif self.mode == 'hier': 
            model = HierModel(config.system, num_classes, config.hier_model, config.layers)
        elif self.mode == 'auto_regressive':
            model = AutoRegressiveModel(config.system, num_classes, config.hier_model, config.layers)
        elif self.mode == 'full_context': 
            model = SpanModel(config.system, num_classes)
        return model 
    
    def model_output(self, batch, train=True):
        if self.mode in ['independent', 'context', 'hier']: 
            y = self.act_model(batch.ids, batch.mask)
        elif self.mode in ['full_context']: 
            y = self.act_model(batch.ids, batch.info)
        elif self.mode in ['auto_regressive']: 
            if train: y = self.act_model(batch.ids, batch.mask, batch.labels)
            else:     y = self.act_model.decode(batch.ids, batch.mask)
        return y
    
    ####################  Methods For Dialogue Act Classification  ####################
    def train_act(self, config):
        self.L.save_config('act_train_cfg', config)

        data_set = self.select_data(config, 'train')
        self.save_act_info(config)
        self.to(self.device)

        optimizer = torch.optim.Adam(self.act_model.parameters(), lr=config.lr)
        if config.scheduling:
              SGD_steps = (len(train_data)*self.epochs)/config.bsz
              lambda1 = lambda i: 10*i/SGD_steps if i <= SGD_steps/10 else 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
              scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

        best_metric = 0
        for epoch in range(config.epochs):
            logger = np.zeros(3)

            for k, batch in enumerate(self.B.act_batches(data_set, config.bsz), start=1):
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
                
            preds, labels = self.act_eval(config, mode='dev')
            decision = np.argmax(preds, axis=-1)
            acc = sum(decision==labels)/len(decision)
            
            if acc > best_metric:
                self.save_act_model('best_epoch')
                best_metric = acc
            self.L.log(acc)
            
        self.load_act_model('best_epoch')
    
    @no_grad
    def act_eval(self, config, mode='test'):
        data_set = self.select_data(config, mode)
        self.to(self.device)
        
        predicted_probs, labels = [], []
        for k, batch in enumerate(self.B.act_batches(data_set), start=1):
            y = self.model_output(batch, train=False)
            loss = self.cross_loss(y, batch.labels)
            pred_prob = F.softmax(y, dim=-1)
            predicted_probs += pred_prob.cpu().tolist()
            labels += batch.labels.cpu().tolist()
        return(predicted_probs, labels)

    ####################  Methods For Act Level Segmentation  ####################
    def train_seg(self, config):
        self.L.save_config('seg_train_cfg', config)
        data_set = self.select_data(config, 'train')
        self.to(self.device)
        optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=1e-5)

        logger = np.zeros(6)
        for k, batch in enumerate(self.B.seg_batches(data_set), start=1):
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
    def seg_eval(self, config, mode='test'):
        data_set = self.select_data(config, mode)
        self.to(self.device)
    
        for k, batch in enumerate(self.S.batches(data_set), start=1):
            y = seg_model(batch.ids, batch.mask)
            loss = torch.mean(self.BCE_loss(y, batch.labels)[batch.mask==1])
            
            if k%100 == 0:
                self.L.log(f'{k:<5}  acc:{logger[0]/logger[1]:.3f}   p:{logger[2]/logger[3]:.3f}   r:{logger[2]/logger[4]:.3f}')
                logger = np.zeros(5)

    ########  Methods For Cascade Segmentation and Act Classification ########
    @no_grad
    def cascade_act_pred(self, config, mode='train'):
        seg_probs = self.turn_seg_probs(config, mode)
        ids, _align, _acts, turn_len, _ = self.turn_seg(seg_probs)
        
        output = []
        for conv in self.B.cascade_act(ids, _align, _acts):
            conv_act_pred = []
            for batch in conv:
                y = self.act_model(batch.ids, batch.mask)
                pred = torch.argmax(y, dim=-1)
                conv_act_pred += pred.cpu().tolist()
            output.append(conv_act_pred)
        
        return ids, output, turn_len
    
    @no_grad
    def cascade_act_eval(self, config, mode='test', thresh=0.5):
        seg_probs = self.turn_seg_probs(config, mode)
        ids, align, acts, _, perf = self.turn_seg(seg_probs, thresh)
        
        self.L.log(f'TURN SEG     P:{perf[0]:.3f}     R:{perf[1]:.3f}')

        predicted_probs, labels = [], []
        for conv in self.B.cascade_act(ids, align, acts):
            for batch in conv:
                y = self.act_model(batch.ids, batch.mask)
                pred_prob = F.softmax(y, dim=-1)
                predicted_probs += pred_prob.cpu().tolist()
                labels += batch.labels
        return predicted_probs, labels
    
    def exact_seg_eval(self, config, mode='test'):
        seg_probs = self.turn_seg_probs(config, mode)
        P, R, T = [], [], []
        
        for thresh in np.arange(0, 1.00, 0.01):
            _, _, _, _, (p, r) = self.turn_seg(seg_probs, thresh)
            if p!=0 or r!=0:
                P.append(p), R.append(r), T.append(thresh)

        plt.plot(R, P)
        plt.xlabel('Recall', size=14)
        plt.ylabel('Precision', size=14)

        for p, r, t in zip(P[::25], R[::25], T[::25]):
            plt.scatter(r, p, marker='x', color='r')
            plt.text(r, p, round(t,2), color='r')          
        
        F1 = [((2*p*r)/(p+r), p, r, t) for p, r, t in zip(P, R, T)]
        op = max(F1)
        print(f'operating point: {op[3]:.2f}   P: {op[1]:.3f}   R: {op[2]:.3f}  F1: {op[0]:.3f}')
        return max(F1)[1]
    
    def turn_seg(self, turn_data, thresh=0.5):
        pairs = lambda x: [(x[i], x[i+1]) for i in range(len(x)-1)]
        
        ids, align, acts, turn_len, cases = [], [], [], [], 0
        for conv in turn_data:
            conv_ids, conv_align, conv_acts, conv_turn_len = [], [], [], []
            for turn in conv:
                decisions = [1] + list((turn.probs > thresh).nonzero()[0])
                segments = pairs(decisions)
                for start, end in segments:
                    segment = turn.ids[start:end]
                    conv_ids.append([101] + segment.tolist() + [102])
                conv_turn_len.append(len(segments))
                
                if turn.segs:  
                    seg_labels = pairs(turn.segs)
                    for start, end in segments:
                        conv_align.append((start, end) in seg_labels)
                        if conv_align[-1]: conv_acts.append(turn.act_dict[end])
                        else:              conv_acts.append(-1)
                    cases += len(seg_labels)
            
            ids.append(conv_ids), align.append(conv_align), acts.append(conv_acts), turn_len.append(conv_turn_len)

        if turn.segs:
            hits = sum([sum(x) for x in align])
            preds = sum([len(x) for x in align])
            if (preds != 0) and (hits != 0): perf = (hits/preds, hits/cases)
            else:                            perf = (0, 0)
            
        return ids, align, acts, turn_len, perf
    
    @no_grad
    def turn_seg_probs(self, config, mode='test'):  
        data_set = self.select_data(config, mode)
        self.to(self.device)

        output = []
        for conv in self.B.cascade_seg(data_set):
            conv_turns = []
            for batch in conv:
                y = self.seg_model(batch.ids, batch.mask).cpu().numpy()
                for ids, mask, probs, segs, label, act_dict \
                in zip(batch.ids, batch.mask, y, batch.segs, batch.labels, batch.act_dict):
                    mask_len = int(sum(mask))
                    ids, probs = ids[:mask_len], probs[:mask_len]
                    turn = {'ids':ids, 'probs':probs, 'segs':segs, 'label':label, 'act_dict':act_dict}       
                    conv_turns.append(SimpleNamespace(**turn))
            output.append(conv_turns)
        return output

    #############    GENERAL UTILITIES     #############
    def select_data(self, config, mode):
        D = ConvHandler(config.data_src, self.system, config.punct, config.action, config.lim, config.class_reduct)
        if   mode == 'train': data_set = D.train 
        elif mode ==   'dev': data_set = D.dev 
        elif mode ==  'test': data_set = D.test
        return data_set

    def save_act_info(self, config):
        D = ConvHandler(config.data_src, self.system, config.punct, config.action, config.lim, config.class_reduct)

        self.tokenizer = D.tokenizer
        self.label_dict = D.label_dict
        self.label_dict[-1] = 'missed'
        
    def save_models(self, name='base'):
        self.save_act_model(name)
        self.save_seg_model(name)

    def save_act_model(self, name='base'):
        device = next(self.act_model.parameters()).device
        self.act_model.to("cpu")
        torch.save(self.act_model.state_dict(), f'{self.L.path}/models/{name}_act.pt')

    def save_seg_model(self, name='base'):
        device = next(self.seg_model.parameters()).device
        self.seg_model.to("cpu")
        torch.save(self.seg_model.state_dict(), f'{self.L.path}/models/{name}_seg.pt')
    
    def load_models(self, name='base'):
        self.load_act_model(name)
        self.load_seg_model(name)

    def load_act_model(self, name='base'):
        self.act_model.load_state_dict(torch.load(self.L.path + f'/models/{name}_act.pt'))
        
    def load_seg_model(self, name='base'):
        self.seg_model.load_state_dict(torch.load(self.L.path + f'/models/{name}_seg.pt'))

    def to(self, device):
        self.act_model.to(device)
        self.seg_model.to(device)
        self.B.to(device)
    
