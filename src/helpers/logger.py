import os
import json
import torch
import shutil

BASE_DIR = '/home/alta/Conversational/OET/al826/2021/dialogue_acts/results'

class Logger:
    def __init__(self, system_cfg, exp_name=None):
        if system_cfg.load:
            pass
        else:
            if system_cfg.save:
                self.exp_name = exp_name
                self.dir = self.get_dir(exp_name)
            else:
                self.dir = self.temp_dir()
                
            self.make_dir(self.dir)
            self.save_config('system_cfg', system_cfg)
            self.log = self.make_logger()
            self.record = self.make_logger(record=True)
            
    def temp_dir(self):
        temp_dir = f'{BASE_DIR}/temp'
        if os.path.isdir(temp_dir): shutil.rmtree(temp_dir)
        return temp_dir
        
    def get_dir(self, exp_name):
        return f'{BASE_DIR}/{exp_name}'
       
    def make_dir(self, dir_):
        os.makedirs(dir_)
        os.mkdir(f'{dir_}/models')

    def save_config(self, config_name, config):
        config_path = f'{self.dir}/{config_name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(config.__dict__, jsonFile, indent=4)

    def make_logger(self, record=False):
        file_name = 'record' if record else 'log'
        log_path = f'{self.dir}/{file_name}.txt'
        open(log_path, 'x').close()  
        
        def log(*x):
            if record: self.log(*x)
            else: print(*x)
                
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
    
    def save_model(self, name, model):
        device = next(model.parameters()).device
        model.to("cpu")
        torch.save(model.state_dict(), f'{self.dir}/models/{name}.pt')
        model.to(device)
    
    @property
    def path(self):
        return self.dir