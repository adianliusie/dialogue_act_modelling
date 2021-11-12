from transformers import BertTokenizerFast, RobertaTokenizerFast, BigBirdTokenizer, ReformerTokenizerFast

from types import SimpleNamespace
import json
from tqdm import tqdm 
import string
import re 

BASE_DIR = '/home/alta/Conversational/OET/al826/2021/dialogue_acts/swbd_data'

def load_json(path):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

class ConvHandler:
    def __init__(self, system=None, punct=True, action=True, debug=False):
        train, dev, test = self.get_act_data()
        train, dev, test = [list(i.values()) for i in [train, dev, test]]
        
        if debug:  train, dev, test = train[:50], dev[:50], test[:50]
            
        Utterance.set_punct(punct)
        Utterance.set_action(action)
        Utterance.set_system(system)
        
        self.train = [Conversation(conv) for conv in tqdm(train)]
        self.dev = [Conversation(conv) for conv in dev]
        self.test = [Conversation(conv) for conv in test]
        self.act_id_dict = load_json(f'{BASE_DIR}/swbd_act_dict.json')
        self.act_names_dict = load_json(f'{BASE_DIR}/swbd_act_names.json')

    def get_act_data(self):
        paths = [f'{BASE_DIR}/swbd_act_{i}.json' for i in ['train', 'dev', 'test']]
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
        turn = {'text':'', 'ids':[], 'acts':[], 'spkr':None}

        for utt in self.utts:
            if utt.spkr != prev_speaker:
                if len(turn['ids'])>0:
                    turn['ids'] = [Utterance.tokenizer.cls_token_id] \
                                + turn['ids'] + [Utterance.tokenizer.sep_token_id]
                if len(turn['text'])>0:
                    turns.append(SimpleNamespace(**turn)) 
                turn = {'text':'', 'ids':[], 'acts':[], 'spkr':utt.spkr}
                
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
    act_id_dict = load_json(f'{BASE_DIR}/swbd_act_dict.json')
    act_names_dict = load_json(f'{BASE_DIR}/swbd_act_names.json')
    tokenizer, punct, action = None, True, True

    def __init__(self, text, spkr, act=None):
        self.text = ' '.join(text).strip()
        if not self.action:
            self.text = re.sub("<<.*?>>", "", self.text).strip()
            self.text = re.sub("<.*?>", "", self.text).strip()

        if not self.punct: 
            self.text = re.sub(r'[^\w\s]', '', self.text).lower().strip()

        self.spkr = spkr
        
        if hasattr(self, 'tokenizer') and self.tokenizer != None:
            self.ids = self.tokenizer(self.text).input_ids
        else:
            self.ids = [None]
            
        if act != None:
            self.act = self.act_id_dict[act]
            self.act_str = self.act_names_dict[act]
            
    @classmethod
    def set_punct(cls, value):
        cls.punct = value
    
    @classmethod
    def set_action(cls, value):
        cls.action = value
    
    @classmethod
    def set_system(cls, system):
        if system in ['bert', 'electra']: 
            cls.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        if system == 'bert_cased':
            cls.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        elif system == 'roberta': 
            cls.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        elif system == 'big_bird': 
            cls.tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
        elif system == 'reformer': 
            cls.tokenizer = ReformerTokenizerFast.from_pretrained('google/reformer-crime-and-punishment')

    def __repr__(self):
        return self.text
