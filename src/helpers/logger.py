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
            system_cfg = self.system_cfg 
            self.log = self.make_logger(file_name='log_2')
        else:
            if not system_cfg.save:
                self.exp_name = 'temp'
                self.del_temp_dir()
            else:
                self.exp_name = system_cfg.exp_name
                
            self.make_dir()
            self.save_config('system_cfg', system_cfg) 
            self.log = self.make_logger()
            self.record = self.make_logger(record=True)

    def del_temp_dir(self):
        if os.path.isdir(f'{BASE_DIR}/temp'): shutil.rmtree(f'{BASE_DIR}/temp')        

    def make_dir(self):
        os.makedirs(self.path)
        os.mkdir(f'{self.path}/models')

    def save_config(self, config_name, config):
        config_path = f'{self.path}/{config_name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(config.__dict__, jsonFile, indent=4)

    def make_logger(self, record=False, file_name=None):
        if file_name is None:
            file_name = 'record' if record else 'log'
            
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
                       
    @property
    def system_cfg(self):
        system_cfg = load_json(f'{self.path}/system_cfg.json')
        return SimpleNamespace(**system_cfg)

    @property
    def path(self):
        return f'{BASE_DIR}/{self.exp_name}'