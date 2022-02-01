import os
import json
import torch
import shutil

from types import SimpleNamespace

from ..utils import load_json

BASE_DIR = '/home/alta/Conversational/OET/al826/2021/dialogue_acts/results'

class Logger:
    def __init__(self, system_cfg):
        if system_cfg.load:
            self.exp_name = system_cfg.exp_name
            self.system_cfg = system_cfg 
            self.log = self.make_logger(file_name='log_2')
        else:
            if not system_cfg.save:
                self.exp_name = 'temp'
                self.del_temp_dir()
            else:
                self.exp_name = system_cfg.exp_name
                
            self.make_dir()
            self.save_config('system_cfg', system_cfg) 
            self.log = self.make_logger(file_name='log')
            self.record = self.make_logger(file_name='record', record=True)

    def del_temp_dir(self):
        if os.path.isdir(f'{BASE_DIR}/temp'): shutil.rmtree(f'{BASE_DIR}/temp')        

    def make_dir(self):
        os.makedirs(self.path)
        os.mkdir(f'{self.path}/models')

    def save_config(self, name:str, config:SimpleNamespace):
        config_path = f'{self.path}/{name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(config.__dict__, jsonFile, indent=4)

    def make_logger(self, file_name, record=False):     
        log_path = f'{self.path}/{file_name}.txt'
        open(log_path, 'a+').close()  
        
        def log(*x):
            if record: self.log(*x)
            else: print(*x)
                
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
                       
    def load_cfg(self, name:str='system_cfg'):
        cfg = load_json(f'{self.path}/{name}.json')
        return SimpleNamespace(**cfg)

    @property
    def path(self):
        return f'{BASE_DIR}/{self.exp_name}'