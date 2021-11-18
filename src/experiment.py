import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_curve

from .helpers import ConvHandler, BatchHandler, Logger
from .models import FlatTransModel, SpanModel
from .utils import no_grad, toggle_grad, LossFunctions

class ExperimentHandler:
    def __init__(self, system_cfg):
        self.L = Logger(system_cfg)
        self.D = ConvHandler(system_cfg.data_src, system_cfg.system, system_cfg.punct, system_cfg.action, system_cfg.debug)
        self.B = BatchHandler(system_cfg.mode, system_cfg.mode_arg, system_cfg.max_len)
        
        if system_cfg.mode == 'full_context':
            self.model = SpanModel(system_cfg.system, len(self.D.act_id_dict))
        else:
            self.model = FlatTransModel(system_cfg.system, len(self.D.act_id_dict))

        self.device = torch.device('cuda') if torch.cuda.is_available() \
                      else torch.device('cpu')
        self.cross_loss = torch.nn.CrossEntropyLoss()

    def train(self, config):
        self.L.save_config('train_cfg', config)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        if config.scheduling:
              SGD_steps = (len(train_data)*self.epochs)/config.bsz
              lambda1 = lambda i: 10*i/SGD_steps if i <= SGD_steps/10 else 1 - ((i - 0.1*SGD_steps)/(0.9*SGD_steps))
              scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

        self.model.to(self.device)
        self.B.to(self.device)
        
        best_metric = 0
        for epoch in range(config.epochs):
            logger = np.zeros(3)

            for k, batch in enumerate(self.B.batches(self.D.train, config.bsz), start=1):
                y = self.model(batch.ids, batch.mask)
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
                    print(f'{k:<5}  {logger[0]/config.print_len:.3f}    {logger[1]/logger[2]:.3f}')
                    logger = np.zeros(3)
                
            self.evaluate(mode='dev')
            self.evaluate(mode='test')

    @no_grad
    def evaluate(self, mode='dev'):
        if   mode ==  'dev': dataset = self.D.dev 
        elif mode == 'test': dataset = self.D.test
        
        predicted_probs, labels = [], []
        for k, batch in enumerate(self.B.batches(dataset), start=1):
            y = self.model(batch.ids, batch.mask)
            loss = self.cross_loss(y, batch.labels)
            pred_prob = F.softmax(y, dim=-1)
            predicted_probs += pred_prob.cpu().tolist()
            labels += batch.labels.cpu().tolist()
        
        return(predicted_probs, labels)

