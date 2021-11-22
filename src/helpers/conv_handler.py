from transformers import BertTokenizerFast, RobertaTokenizerFast, BigBirdTokenizer, ReformerTokenizerFast, AlbertTokenizerFast

from types import SimpleNamespace
import json
from tqdm import tqdm 
import string
import re 

from ..utils import load_json

class ConvHandler:
    def __init__(self, data_src, system=None, punct=True, action=True, debug=False):
        self.set_up_paths(data_src)
        
        train, dev, test = self.get_act_data()
        train, dev, test = [list(i.values()) for i in [train, dev, test]]
        
        if debug:  train, dev, test = train[:10], dev[:10], test[:10]
            
        Utterance.set_punct(punct)
        Utterance.set_action(action)
        Utterance.set_system(system)
        
        self.train = [Conversation(conv) for conv in tqdm(train)]
        self.dev = [Conversation(conv) for conv in dev]
        self.test = [Conversation(conv) for conv in test]

    def set_up_paths(self, data_src):
        global BASE_DIR, act_id_dict, act_names_dict
        if data_src == 'swbd':
            BASE_DIR = '/home/alta/Conversational/OET/al826/2021/dialogue_acts/act_data/swbd_act/data'
        elif data_src == 'ami':
            BASE_DIR = '/home/alta/Conversational/OET/al826/2021/dialogue_acts/act_data/ami_act/data'
        act_id_dict = load_json(f'{BASE_DIR}/act_dict.json')
        act_names_dict = load_json(f'{BASE_DIR}/act_names.json')
        self.act_id_dict = act_id_dict
        
    def get_act_data(self):
        paths = [f'{BASE_DIR}/act_{i}.json' for i in ['train', 'dev', 'test']]
        train, dev, test = [load_json(i) for i in paths]
        return (train, dev, test)
    
class Conversation:
    def __init__(self, data):
        self.data = data
        self.utts = [Utterance(**utt) for utt in self.data]
        self.turns = self.make_turns()
        
    def make_turns(self):
        turns = []
        prev_speaker = None
        turn = {'text':'', 'ids':[], 'segs':[], 'acts':[], 'spkr':None}

        for utt in self.utts:
            if utt.spkr != prev_speaker:                    
                if len(turn['text'])>0:
                    turn['ids'] = [Utterance.tokenizer.cls_token_id] \
                                + turn['ids'] + [Utterance.tokenizer.sep_token_id]
                    turns.append(SimpleNamespace(**turn)) 
                turn = {'text':'', 'ids':[], 'segs':[], 'acts':[], 'spkr':None}
                
            turn['text'] += ' ' + utt.text
            turn['ids'] += utt.ids[1:-1]
            turn['acts'].append(utt.act)
            prev_speaker = utt.spkr
            
        if len(turn['ids'])>0:
            turn['ids'] = [Utterance.tokenizer.cls_token_id] + turn['ids'] \
                        + [Utterance.tokenizer.sep_token_id]
        if len(turn['text'])>0:
            turns.append(SimpleNamespace(**turn)) 
        return turns
    
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

class Utterance:
    tokenizer, punct, action = None, True, True

    def __init__(self, text, spkr, act=None):
        self.text = text
        if not self.action:
            self.text = re.sub("<<.*?>>", "", self.text)
            self.text = re.sub("<.*?>", "", self.text)

        if not self.punct: 
            self.text = re.sub(r'[^\w\s]', '', self.text)

        self.text = self.text.strip()
        self.spkr = spkr
        
        if self.tokenizer != None:
            self.ids = self.tokenizer(self.text).input_ids
        else:
            self.ids = [None]
            
        if act != None:
            self.act = act_id_dict[act]
            self.act_str = act_names_dict[act] if act in act_names_dict else None
            
    @classmethod
    def set_punct(cls, value):
        cls.punct = value
    
    @classmethod
    def set_action(cls, value):
        cls.action = value
    
    @classmethod
    def set_system(cls, system):
        if system in ['bert', 'electra', 'rand']: 
            cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        if system == 'bert_cased':
            cls.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        elif system == 'roberta': 
            cls.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif system == 'big_bird': 
            cls.tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
        elif system == 'albert':
            cls.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
            
    def __repr__(self):
        return self.text
