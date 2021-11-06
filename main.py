from types import SimpleNamespace
from src.experiment import ExperimentHandler 

system_cfg = {'punct':False, 'action':True, 'context':'back_history', 'context_len':5}
train_cfg = {'lr':1e-5, 'scheduling':False, 'epochs':1, 'print_len':200, 'bsz':8}

system_cfg = SimpleNamespace(**system_cfg)
train_cfg = SimpleNamespace(**train_cfg)

E = ExperimentHandler(system_cfg)
E.train(train_cfg)
