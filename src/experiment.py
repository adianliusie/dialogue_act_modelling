import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve

from .helpers import ConvHandler, BatchHandler, Logger
from .models import FlatTransModel, SpanModel, HierModel, AutoRegressiveModel, FlatSegModel
from .utils import no_grad, toggle_grad, LossFunctions

class ExperimentHandler:
    def __init__(self, system_cfg):
        load = system_cfg.load

        self.L = Logger(system_cfg)
        if load: system_cfg = self.L.system_cfg

        self.mode = system_cfg.mode
        self.system = system_cfg.system
        
        self.act_model = self.create_act_model(system_cfg, 43)
        self.seg_model = FlatSegModel(self.system)
        if load: self.load_models()
        self.B = BatchHandler(system_cfg.mode, system_cfg.mode_arg, system_cfg.max_len_conv, system_cfg.max_len_utts)

        self.device = torch.device(system_cfg.device) if torch.cuda.is_available() else torch.device('cpu')

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
                
            preds, labels = self.eval_act(config, mode='dev')
            decision = np.argmax(preds, axis=-1)
            acc = sum(decision==labels)/len(decision)
            
            if acc > best_metric:
                self.save_act_model('best_epoch')
                best_metric = acc
            self.L.log(acc)
            
        self.load_act_model('best_epoch')
    
    @no_grad
    def eval_act(self, config, mode='test'):
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
    def eval_seg(self, config, mode='test'):
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
    def pred_conv_act(self, config, mode='train'):
        seg_ids, _align, _segs, turn_len = self.eval_conv_seg(config, mode)
        
        output = []
        for conv in self.B.cascade_act(seg_ids, _align, _segs):
            conv_act_pred = []
            for batch in conv:
                y = self.act_model(batch.ids, batch.mask)
                pred = torch.argmax(y, dim=-1)
                conv_act_pred += pred.cpu().tolist()
            output.append(conv_act_pred)
        
        return seg_ids, output, turn_len
    
    @no_grad
    def eval_conv_act(self, config, mode='test'):
        ids, align, acts, _ = self.eval_conv_seg(config, mode)
        
        predicted_probs, labels = [], []
        for conv in self.B.cascade_act(ids, align, acts):
            for batch in conv:
                y = self.act_model(batch.ids, batch.mask)
                pred_prob = F.softmax(y, dim=-1)
                predicted_probs += pred_prob.cpu().tolist()
                labels += batch.labels
        return predicted_probs, labels
    
    @no_grad
    def eval_conv_seg(self, config, mode='test'):  
        data_set = self.select_data(config, mode)
        pairs = lambda x: [(x[i], x[i+1]) for i in range(len(x)-1)]
        batch_pair = lambda x: [pairs(seg_change) for seg_change in x]
        
        ids, align, acts, turn_len, utt_count,  = [], [], [], [], 0
        for conv_num, conv in enumerate(self.B.cascade_seg(data_set), start=1):
            conv_ids, conv_align, conv_acts, conv_turn_len = [], [], [], []
            for batch in conv:
                y = self.seg_model(batch.ids, batch.mask)

                # for each turn, determine utterance start/end position
                decisions = [[1] for _ in range(len(y))]
                for i, j in (y>0.5).nonzero(as_tuple=False):
                    if j < batch.mask[i].sum():
                        decisions[i].append(int(j))
                seg_pred = batch_pair(decisions)
                
                for k, turn in enumerate(batch.ids):
                    for start, end in seg_pred[k]:
                        segment = turn[start:end]
                        conv_ids.append([101] + segment.tolist() + [102])
                    conv_turn_len.append(len(seg_pred[k]))
                    
                #Evaluate if labels are available
                if batch.segs:  
                    seg_labels = batch_pair(batch.segs)
                    for k, turn in enumerate(batch.ids):
                        for start, end in seg_pred[k]:
                            conv_align.append((start, end) in seg_labels[k])
                            if conv_align[-1]: conv_acts.append(batch.act_dict[k][end])
                            else:              conv_acts.append(-1)
                    utt_count += sum([1 for turn in seg_labels for utt in turn])

            ids.append(conv_ids), align.append(conv_align), acts.append(conv_acts), turn_len.append(conv_turn_len)

        if batch.segs:
            hits = sum([sum(x) for x in align])
            preds = sum([len(x) for x in align])
            self.L.log(f'SEG  P:{hits/preds:.3f}   R:{hits/utt_count:.3f}')
            
        return ids, align, acts, turn_len

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
    
