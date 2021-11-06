import torch

def no_grad(func):
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner
