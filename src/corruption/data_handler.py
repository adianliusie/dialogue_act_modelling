from types import SimpleNamespace
import json
import copy
import time
import numpy as np

from ..helpers import ConvHandler

def load_json(path):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data
    
class DataHandler():
    def __init__(self, data_src):
        if data_src in ['wsj']:
            self.train, self.dev, self.test = self.get_data_sets(data_src)

        if data_src in ['wiki', 'wiki_small', 'wsj']:
            self.train, self.dev, self.test = self.get_data_sets('wiki')
            
            self.dev = self.dev[:5_000]
            self.test = self.test[:5_000]
            
            if data_src == 'wiki_small':
                self.train = self.train[:100_000]
 
        if data_src in ['wiki_unfiltered', 'yahoo_unlabelled']:
            self.train = self.get_data(data_src)

        if data_src in ['gcdc']:
            data = self.get_data_sets('gcdc')
            train, test = data[:4], data[4:]
            domains = ['clinton', 'enron', 'yelp', 'yahoo']
            
            for domain, data in zip(domains, train):
                train_name = f'{domain}_train'
                dev_name = f'{domain}_dev'
                setattr(self, train_name, data[:900].copy())
                setattr(self, dev_name, data[900:].copy())
    
            for domain, data in zip(domains, test):
                test_name = f'{domain}_test'
                setattr(self, test_name, data)
            
            self.train = self.clinton_train + self.enron_train + self.yelp_train + self.yahoo_train
            self.dev = self.clinton_dev + self.enron_dev + self.yelp_dev + self.yahoo_dev
            self.test = self.clinton_test + self.enron_test + self.yelp_test + self.yahoo_test

        if data_src in ['swbd']:
            D = ConvHandler(punct=False, action=False)
            self.train = [[utt.text for utt in conv.turns] for conv in D.train]
            self.dev = [[utt.text for utt in conv.turns] for conv in D.dev]
            self.test = [[utt.text for utt in conv.turns] for conv in D.test]
            
    def get_data(self, data_src):
        path = self.get_path(data_src)
        dataset = load_json(path) 
        data = self.objectify(dataset)
        return data
    
    def get_data_sets(self, data_src):
        paths = self.get_paths(data_src)
        data = [load_json(path) for path in paths]
        data = [self.objectify(dataset) for dataset in data]
        return data
    
    def objectify(self, data):
        return [SimpleNamespace(**ex) for ex in data]

    def get_path(self, data_src):
        if data_src == 'wiki_unfiltered': 
            path = '/home/alta/Conversational/OET/al826/2021/data/unlabelled/wiki_unfiltered.json'
        if data_src == 'yahoo_unlabelled': 
            path = '/home/alta/Conversational/OET/al826/2021/data/unlabelled/yahoo_unlabelled.json'
        return path 
    
    def get_paths(self, data_src):
        if data_src == 'wiki': paths = self.wiki_paths()
        if data_src == 'wsj':  paths = self.wsj_paths()
        if data_src == 'gcdc':  paths = self.gcdc_paths()
        return paths
    
    def wiki_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/unlabelled'
        paths = [f'{base_dir}/wiki_{i}.json' for i in ['train', 'dev', 'test']]
        return paths
    
    def wsj_paths(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data/coherence/WSJ'
        paths = [f'{base_dir}/WSJ_{i}.json' for i in ['train', 'dev', 'test']]
        return paths

    def gcdc_paths(self):
        base_dir = "/home/alta/Conversational/OET/al826/2021/data/coherence/GCDC"
        paths = []
        for set_ in ['train', 'test']:
            for domain in ['clinton', 'enron', 'yelp', 'yahoo']:
                paths.append(f'{base_dir}/{domain}_{set_}.json')
        return paths

    