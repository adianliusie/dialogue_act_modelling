from types import SimpleNamespace
import json
from tqdm import tqdm 
import string
import re 

from src.utils import load_json, get_tokenizer, flatten

class ConvHandler:
    cache = {}
    def __init__(self, data_src, system='bert', punct=True, action=True, lim=None, class_reduct=False): 
        self.tokenizer = get_tokenizer(system)
        Utterance.set_punct(punct)
        Utterance.set_action(action)
        Utterance.set_system(system)
        
        #use hashing so that if data processed recently, use the processed version
        arg_hash = hash((data_src, system, punct, action, class_reduct))
        if arg_hash in self.cache:
            self.__dict__ = self.cache[arg_hash]
        else:
            self.get_data(data_src, lim)   
            self.__class__.cache[arg_hash] = self.__dict__

    def get_data(self, data_src, lim=None, class_reduct=False):
        if data_src in ['swbd', 'ami']: self.get_train_sets(data_src, lim, class_reduct)
        elif data_src == 'oet'        : self.get_oet_data()
        
    def get_train_sets(self, data_src, lim=None, class_reduct=False):
        base_dir = f'/home/alta/Conversational/OET/al826/2021/data_2/conversations/{data_src}'
        paths = [f'{base_dir}/{i}.json' for i in ['train', 'dev', 'test']]
        train, dev, test = [load_json(i) for i in paths]
        self.create_label_dict(f'{base_dir}/labels.json')
        
        if lim:  train, dev, test = train[:lim], dev[:lim], test[:lim]
        self.train = [Conversation(conv['turns']) for conv in tqdm(train)]
        self.dev   = [Conversation(conv['turns']) for conv in tqdm(dev)] 
        self.test  = [Conversation(conv['turns']) for conv in tqdm(test)] 
        
    def create_label_dict(self, path):
        global label_dict
        label_dict = load_json(path)
        label_dict = {int(k):v for k, v in label_dict.items()}
        self.label_dict = label_dict

    def get_oet_data(self):
        base_dir = '/home/alta/Conversational/OET/al826/2021/data_2/conversations/oet'
        data = load_json(f'{base_dir}/oet_small.json')
        self.train = [Conversation(conv['turns']) for conv in tqdm(data)]

class Conversation:
    def __init__(self, data):
        self.data = data
        
        self.utts  = [Utterance(**utt) for utt in self.data]
        self.turns = self.make_turns()
 
    def make_turns(self):
        turns = []
        prev_speaker = None
        turn = {'text':'', 'speaker':None, 'tags':{'segs':[], 'labels':[]}}
        i = 0
        for utt in self.utts:
            if utt.speaker != prev_speaker:                    
                if len(turn['text'])>0:             
                    turn_obj = Utterance(**turn)
                    turns.append(turn_obj) 
                    assert turn_obj.ids[1:-1] == flatten(turn['tags']['ids'])

                turn = {'text':'', 'speaker':utt.speaker, 'tags':{'segs':[], 'labels':[], 'ids':[]}}
                
            turn['text'] += ' ' + utt.text
            turn['tags']['labels'].append(utt.label)
            turn['tags']['ids'].append(utt.ids[1:-1])
            turn['tags']['segs'].append(len(flatten(turn['tags']['ids']))+1)
            prev_speaker = utt.speaker
            
        if len(turn['text'])>0:
            turns.append(Utterance(**turn)) 
        return turns
    
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

class Utterance:
    punct, action, tokenizer = True, True, get_tokenizer('bert')

    def __init__(self, text, speaker=None, label=None, tags=None):
        self.text = text
        self.clean_text()
        self.speaker = speaker
        self.label = label
        self.tags = tags
        self.ids = self.tokenizer(self.text).input_ids

                    
        if label != None:
            self.label_name = label_dict[label]
            
    def clean_text(self):
        if not self.action:
            self.text = re.sub("<<.*?>>", "", self.text)
            self.text = re.sub("<.*?>", "", self.text)
        if not self.punct: 
            self.text = re.sub(r'[^\w\s]', '', self.text)
        self.text = self.text.strip()
        
    @classmethod
    def set_punct(cls, value):
        cls.punct = value
    
    @classmethod
    def set_action(cls, value):
        cls.action = value
    
    @classmethod
    def set_system(cls, system):
            cls.tokenizer = get_tokenizer(system)

    def __repr__(self):
        return self.text