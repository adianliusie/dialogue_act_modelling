import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

from .helpers import ConvHandler, DActsBatcher, SegBatcher, Logger
from .models import FlatTransModel, SpanModel, HierModel, AutoRegressiveModel, FlatSegModel
from .utils import no_grad, toggle_grad, LossFunctions, pairs

class ExperimentHandler:
    """"base class for running all DA classification experiments"""
    
    def __init__(self, system_cfg:SimpleNamespace):
        self.device = torch.device(system_cfg.device) if torch.cuda.is_available() else torch.device('cpu')
        self.L = Logger(system_cfg)
        
        if system_cfg.load: self.load_models()

        self.cross_loss = torch.nn.CrossEntropyLoss()
        self.BCE_loss = torch.nn.BCELoss(reduction='none')
    
    
    ####################  Methods For Dialogue Act Classification  ####################
    
    def train_act(self, config:SimpleNamespace):
        """trains a DA classification model. Given phrases, learns DA class. 
           Saves model in log dir and loads best epoch in self.act_model"""
        
        self.mode = config.mode 
        self.act_model = self.create_act_model(config, 43)
        self.act_batcher = DActsBatcher(config.mode, config.mode_arg, config.max_len)
        self.L.save_config('act_cfg', config)

        data_set = self.select_data(config, 'train')
        self.to(self.device)

        optimizer = torch.optim.Adam(self.act_model.parameters(), lr=config.lr)
        if config.scheduling:
              SGD_steps = (len(train_data)*self.epochs)/config.bsz
              lambda1 = lambda i: 10*i/SGD_steps if i <= SGD_steps/10 else 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
              scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

        best_metric = 0
        for epoch in range(config.epochs):
            print('epoch', ' '*4, 'steps',' '*4, 'loss', ' '*5, 'acc',)
            logger = np.zeros(3)

            for k, batch in enumerate(self.act_batcher.batches(data_set, config.bsz, shuf=True), start=1):
                #forward and loss calculation
                y = self.model_output(batch)
                loss = self.cross_loss(y, batch.labels)

                #updating model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if config.scheduling: scheduler.step()
                
                #basic performance logging
                y_pred = torch.argmax(y, -1)
                logger[0] += loss.item()
                logger[1] += sum(y_pred == batch.labels)
                logger[2] += len(batch.labels)

                #print every now and then
                if k%config.print_len == 0:
                    self.L.log(f'  {epoch:<3}      {k:<5}     {logger[0]/config.print_len:.3f}      {logger[1]/logger[2]:.3f}')
                    logger = np.zeros(3)
                
            preds, labels, loss = self.act_eval(config, mode='dev')
            decision = np.argmax(preds, axis=-1)
            acc = sum(decision==labels)/len(decision)
            
            self.L.log(f'\n DEV {epoch:<3}   loss:{loss:.3f}   acc:{acc:.3f}\n')
            self.L.log('#'*40 + '\n')

            if acc > best_metric:
                self.save_act_model()
                best_metric = acc
            
        self.load_act_model()
        
    def create_act_model(self, config:SimpleNamespace, num_classes:int)->'model':
        """creates DA model and randomly initialises it"""
        
        if config.mode in ['independent', 'context']: 
            model = FlatTransModel(config.system, num_classes)
        elif config.mode == 'hier': 
            model = HierModel(config.system, num_classes, config.hier_model, config.layers)
        elif config.mode == 'auto_regressive':
            model = AutoRegressiveModel(config.system, num_classes, config.hier_model, config.layers)
        elif self.mode == 'full_context': 
            model = SpanModel(config.system, num_classes)
        return model
    
    def model_output(self, batch:SimpleNamespace, train=True)->torch.Tensor:
        """Sends input tensors to model and returns DA predictions"""

        if self.mode in ['independent', 'context', 'hier']: 
            y = self.act_model(batch.ids, batch.mask)
        elif self.mode in ['full_context']: 
            y = self.act_model(batch.ids, batch.info)
        elif self.mode in ['auto_regressive']: 
            if train: y = self.act_model(batch.ids, batch.mask, batch.labels)
            else:     y = self.act_model.decode(batch.ids, batch.mask)
        return y
    
    @no_grad
    def act_eval(self, config, mode:str='test')->Tuple[list, list, int]:
        """Returns predictions and labels for DA classification"""
 
        data_set = self.select_data(config, mode)
        self.to(self.device)
        
        predicted_probs, labels, loss = [], [], 0
        for k, batch in enumerate(self.act_batcher.batches(data_set), start=1):
            y = self.model_output(batch, train=False)
            loss += self.cross_loss(y, batch.labels)
            pred_prob = F.softmax(y, dim=-1)
            predicted_probs += pred_prob.cpu().tolist()
            labels += batch.labels.cpu().tolist()
        return(predicted_probs, labels, loss/k)

    
    
    ####################  Methods For Act Level Segmentation  ####################
    
    def train_seg(self, config):
        """trains turn segmentation model to phrases"""
        
        self.seg_model = FlatSegModel(config.system)
        self.seg_batcher = SegBatcher(config.max_len)
        self.L.save_config('seg_cfg', config)

        data_set = self.select_data(config, 'train')
        self.to(self.device)

        optimizer = torch.optim.Adam(self.seg_model.parameters(), lr=config.lr)

        logger = np.zeros(6)
        for epoch in range(config.epochs):
            for k, batch in enumerate(self.seg_batcher.batches(data_set, config.bsz, shuf=True), start=1):
                #forward
                y = self.seg_model(batch.ids, batch.mask)
                loss = torch.mean(self.BCE_loss(y, batch.labels)[batch.mask==1])

                #updating model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #basic performance logging
                logger += self.seg_batch_perf(y, batch.mask, batch.labels, loss.item())

                if k%config.print_len == 0:
                    self.L.log(f'{k:<5}  loss:{logger[5]/100:.3f}  acc:{logger[0]/logger[1]:.3f}  '  
                               f'p:{logger[2]/logger[3]:.3f}  r:{logger[2]/logger[4]:.3f}')
                    logger = np.zeros(6)

            #need to eval on dev set
            self.save_seg_model()
        self.load_seg_model()

    @staticmethod
    def seg_batch_perf(y, mask, labels, loss):
        """processes batch prediction and labels to give TP etc."""
        
        result = np.zeros(6)
        result[0] = ((y>0.5)==labels)[mask==1].sum()
        result[1] = mask.sum()                             
        result[2] = (y>0.5)[mask*labels==1].sum()    
        result[3] = (y>0.5).sum()
        result[4] = (mask*labels).sum()
        result[5] = loss
        return result
    
    @no_grad
    def seg_eval(self, config, mode='test'):
        data_set = self.select_data(config, mode)
        self.to(self.device)
    
        logger, loss = np.zeros(6), 0 
        for k, batch in enumerate(self.seg_batcher.batches(data_set, shuf=False), start=1):
            y = self.seg_model(batch.ids, batch.mask)
            loss += torch.mean(self.BCE_loss(y, batch.labels)[batch.mask==1])
            logger += self.seg_batch_perf(y, batch.mask, batch.labels, loss.item())

        self.L.log(f'       acc:{logger[0]/logger[1]:.3f}   p:{logger[2]/logger[3]:.3f}   r:{logger[2]/logger[4]:.3f}')
    
    @no_grad
    def seg_PR_curve(self, config, mode='test'):
        turn_preds = []
        data_set = self.select_data(config, mode)

        
        #Do forward for all turns and save phrases boundary probabilities
        for k, batch in enumerate(self.seg_batcher.batches(data_set, shuf=False), start=1):
            y = self.seg_model(batch.ids, batch.mask).cpu().numpy()
            for probs, lab in zip(y, batch.segs):
                turn_preds.append((probs, lab))
        
        #for each threshold find phrase segmentation performance  
        P,R,T = [],[],[]
        for thresh in np.arange(0, 1.00, 0.01):
            logger = np.zeros(3)
            for probs, labels in turn_preds:
                decisions = [1] + list((probs > thresh).nonzero()[0])
                pred_phrases = set(pairs(decisions))
                ref_phrases  = set(pairs(labels))
                logger[0] += len(pred_phrases.intersection(ref_phrases))
                logger[1] += len(pred_phrases)
                logger[2] += len(ref_phrases)

            p, r = logger[0]/logger[1], logger[0]/logger[2]
            if p or r: P.append(p), R.append(r), T.append(thresh)

        self.plot_PR(R, P, T)
            
    @staticmethod
    def plot_PR(R, P, T):
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

                
    ########  Methods For Cascade Segmentation and Act Classification ########
    
    @no_grad
    def cascade_pred(self, config, mode='test'):
        """takes turns of conversations, creates phrases, and then outputs DA sequence"""
        data_set = self.select_data(config, mode)
        self.to(self.device)

        output = {}
        for k, (conv_id, conv_batch) in enumerate(self.seg_batcher.conv_batches(data_set), start=1):
            phrases = self.seg_model.get_phrases(conv_batch.ids, conv_batch.mask)
            ctx_phrases = self.act_batcher.prep_conv_ids(phrases)
            ids, mask = self.act_batcher.get_padded_ids(ctx_phrases)
            y = self.act_model(ids, mask)
            DA_decisions = torch.argmax(y, dim=-1)
            output[conv_id] = DA_decisions
        return output 
            
    @no_grad
    def cascade_act_eval(self, config, mode='test', thresh=0.5):
        seg_output = self.turn_seg_probs(config, mode)
        ids, align, acts, _, perf = self.turn_seg(seg_output, thresh)
        self.L.log(f'TURN SEG     P:{perf[0]:.3f}     R:{perf[1]:.3f}')
            
        #super unreadable but gets the act labels for all convs
        labels = [ [v for turn in conv for k, v in sorted(turn.act_dict.items())] 
                      for conv in seg_output]
        
        predicted_probs, align_labels = [], []
        for conv in self.B.cascade_act(ids, align, acts):
            act_pred, conv_lab = [], []
            for batch in conv:
                y = self.act_model(batch.ids, batch.mask)
                pred_prob = F.softmax(y, dim=-1)
                act_pred += pred_prob.cpu().tolist()
                conv_lab += batch.labels
            predicted_probs.append(act_pred), align_labels.append(conv_lab)
        return predicted_probs, labels, align_labels
    
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
    
    @no_grad
    def turn_seg_probs(self, config, mode='test'):  
        data_set = self.select_data(config, mode)
        self.to(self.device)

        output = []
        for conv in self.B.cascade_seg(data_set):
            conv_turns = []
            for batch in conv:
                #predict phrase starts using segmentation model
                y = self.seg_model(batch.ids, batch.mask).cpu().numpy()
                for ids, mask, probs, segs, label, act_dict \
                in zip(batch.ids, batch.mask, y, batch.segs, batch.labels, batch.act_dict):
                    mask_len = int(sum(mask))
                    ids, probs = ids[:mask_len], probs[:mask_len]
                    turn = {'ids':ids, 'probs':probs, 'segs':segs, 'seg_label':label, 'act_dict':act_dict}       
                    conv_turns.append(SimpleNamespace(**turn))
            output.append(conv_turns)
        return output
    
    # processes segmentation predictions to return phrases, whether it matches, and act labels
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

    #############    GENERAL UTILITIES     #############
    def select_data(self, cfg, mode):
        D = ConvHandler(data_src=cfg.data_src, system=cfg.system, class_reduct=cfg.class_reduct, 
                        punct=cfg.punct, action=cfg.action, pre_proc=cfg.pre_proc, lim=cfg.lim)
            
        if   mode == 'train': data_set = D.train 
        elif mode ==   'dev': data_set = D.dev 
        elif mode ==  'test': data_set = D.test
        return data_set

    def get_label_names(self, cfg):
        D = ConvHandler(data_src=cfg.data_src, system=cfg.system, class_reduct=cfg.class_reduct, 
                        punct=cfg.punct, action=cfg.action, pre_proc=cfg.pre_proc, lim=cfg.lim)
        if cfg.class_reduct:
            label_dict = D.label_dict
        else:
            label_dict = D.label_dict
        
        return label_dict
    
    #############   SAVING AND LOADING    #############
    def save_act_info(self, config):
        D = ConvHandler(data_src=cfg.data_src, system=cfg.system, class_reduct=cfg.class_reduct, 
                        punct=cfg.punct, action=cfg.action, pre_proc=cfg.pre_proc, lim=cfg.lim)

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
        self.act_model.to(self.device)

    def save_seg_model(self, name='base'):
        device = next(self.seg_model.parameters()).device
        self.seg_model.to("cpu")
        torch.save(self.seg_model.state_dict(), f'{self.L.path}/models/{name}_seg.pt')
        self.seg_model.to(self.device)

    def load_models(self, name='base'):
        """loads models in experiment in directory"""
        self.load_act_model(name)
        self.load_seg_model(name)

    def load_act_model(self, name='base'):
        config = self.L.load_cfg('act_cfg')
        self.act_model = self.create_act_model(config, 43)
        self.act_model.load_state_dict(torch.load(self.L.path + f'/models/{name}_act.pt'))

        self.act_batcher = DActsBatcher(config.mode, config.mode_arg, config.max_len)
        self.mode = config.mode 

    def load_seg_model(self, name='base'):
        config = self.L.load_cfg('seg_cfg')
        self.seg_model = FlatSegModel(config.system)
        self.seg_model.load_state_dict(torch.load(self.L.path + f'/models/{name}_seg.pt'))
        self.seg_batcher = SegBatcher(config.max_len) ##Fill later
        
    def to(self, device):
        if hasattr(self, 'act_model'):   self.act_model.to(device)
        if hasattr(self, 'seg_model'):   self.seg_model.to(device)
        if hasattr(self, 'act_batcher'): self.act_batcher.to(device)
        if hasattr(self, 'seg_batcher'): self.seg_batcher.to(device)

