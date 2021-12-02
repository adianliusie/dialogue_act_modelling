import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve

from .helpers import ConvHandler, BatchHandler, Logger
from .models import FlatTransModel, SpanModel, HierModel, AutoRegressive
from .utils import no_grad, toggle_grad, LossFunctions

class ExperimentHandler:
    def __init__(self, system_cfg):
        self.L = Logger(system_cfg) 
        if system_cfg.load: system_cfg = self.L.system_cfg
        self.D = ConvHandler(system_cfg.data_src, system_cfg.system, system_cfg.punct, system_cfg.action, system_cfg.debug, 
                             system_cfg.class_reduct)
        self.B = BatchHandler(system_cfg.mode, system_cfg.mode_arg, system_cfg.max_len)

        self.mode = system_cfg.mode
        num_classes = max(self.D.act_id_dict.values())+1
        if self.mode in ['independent', 'back_history', 'context']:
            self.model = FlatTransModel(system_cfg.system, num_classes)
        elif self.mode == 'hier':
            self.model = HierModel(system_cfg.system, num_classes, system_cfg.decoder, system_cfg.layers)
        elif self.mode == 'full_context':
            self.model = SpanModel(system_cfg.system, num_classes)
        elif self.mode == 'auto_regressive':
            self.model = AutoRegressive(system_cfg.system, num_classes)

        self.device = torch.device(system_cfg.device) if torch.cuda.is_available() \
                      else torch.device('cpu')
        
        self.cross_loss = torch.nn.CrossEntropyLoss()

    def model_output(self, batch, train=True):
        if self.mode in ['independent', 'back_history', 'context', 'hier']: 
            y = self.model(batch.ids, batch.mask)
        elif self.mode in ['full_context']: 
            y = self.model(batch.ids, batch.info)
        elif self.mode in ['auto_regressive']: 
            if train: y = self.model(batch.ids, batch.mask, batch.labels)
            else:     y = self.model.decode(batch.ids, batch.mask)

        return y
    
    def train(self, config):
        self.to_device()

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
        self.to_device()

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

    def save_model(self, name):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), f'{self.L.path}/models/{name}.pt')
        self.model.to(device)

    def load_model(self, name):
        self.model.load_state_dict(torch.load(self.L.path + f'/models/{name}.pt'))

    def to_device(self):
        self.model.to(self.device)
        self.B.to(self.device)
        