import torch

def no_grad(func):
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner

def toggle_grad(func):
    def inner(*args, no_grad=False):
        if no_grad==True:
            with torch.no_grad():
                return func(*args)
        else:
            return func(*args)
    return inner

class LossFunctions:
    @staticmethod
    def ranking(y1, y2):
        log_likelihood = logsigmoid(y1-y2)
        loss =  -1*torch.mean(log_likelihood)
        return loss
    
    @staticmethod
    def classification(y1, y2):
        log_likelihood = logsigmoid(y1) + logsigmoid(-1*y2)
        loss =  -1*torch.mean(log_likelihood)/2
        return loss       
    
    @staticmethod
    def cross_loss(y, targets):
        loss = cross_entropy(y, targets)
        return loss
    
    @staticmethod
    def MSE(y, targets):
        loss = mse_loss(y, targets)
        return loss
    
