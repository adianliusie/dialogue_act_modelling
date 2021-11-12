import torch
import numpy as np

from .helpers import ConvHandler, BatchHandler
from .models import FlatTransModel, SpanModel
from .utils import no_grad, toggle_grad, LossFunctions

class ExperimentHandler:
    def __init__(self, system_cfg):        
        self.D = ConvHandler(system_cfg.system, system_cfg.punct, system_cfg.action, system_cfg.debug)
        self.B = BatchHandler(system_cfg.mode, system_cfg.mode_arg, system_cfg.max_len)
        
        if system_cfg.mode == 'full_context':
            self.model = SpanModel(system_cfg.system, len(self.D.act_id_dict))
        else:
            self.model = FlatTransModel(system_cfg.system, len(self.D.act_id_dict))

        self.device = torch.device('cuda') if torch.cuda.is_available() \
                      else torch.device('cpu')
        self.cross_loss = torch.nn.CrossEntropyLoss()

    def train(self, config):
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
        
        confusion_matrix = np.zeros([len(self.D.act_id_dict), len(self.D.act_id_dict)])
        for k, batch in enumerate(self.B.batches(dataset), start=1):
            y = self.model(batch.ids, batch.mask)
            loss = self.cross_loss(y, batch.labels)
            
            y_pred = torch.argmax(y, -1)
            
            for pred, lab in zip(y_pred, batch.labels):
                confusion_matrix[lab, pred] += 1
        
        accuracy, precision, recall, label_dist = self.proc_matrix(confusion_matrix)
        macro_F1, micro_F1 = self.get_F1(precision, recall, label_dist)
        print(f'acc: {accuracy:.3f}  macro F1 {macro_F1:.3f}  micro F1 {micro_F1:.3f}')
        return (accuracy, macro_F1, micro_F1)

    def proc_matrix(self, matrix):
        true_positives = np.diagonal(matrix)
        label_count    = np.sum(matrix, axis=1)
        pred_count     = np.sum(matrix, axis=0)
            
        accuracy = sum(true_positives)/sum(label_count)
        precision = true_positives/pred_count
        recall = true_positives/label_count
        label_dist = label_count/np.sum(label_count)
        return (accuracy, precision, recall, label_dist)

    def get_F1(self, precision, recall, label_dist):
        F1 = 2*(precision*recall)/(precision+recall)
        macro_F1 = np.mean(F1[~np.isnan(F1)])
        micro_F1 = np.nansum(label_dist*F1)
        return (macro_F1, micro_F1)
    