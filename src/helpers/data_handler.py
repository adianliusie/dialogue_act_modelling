from transformers import BertTokenizerFast
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

class DataHandler:
    def __init__(self, punct=True, action=True):
        train, dev, test = self.get_act_data()
        
        Utterance.set_punct(punct)
        Utterance.set_action(action)

        self.train = [Conversation(conv) for conv in tqdm(train.values())]
        self.dev = [Conversation(conv) for conv in dev.values()]
        self.test = [Conversation(conv) for conv in test.values()]
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
            if utt.spkr == prev_speaker:
                turn['text'] += ' ' + utt.text
                turn['ids'] += utt.ids[1:-1]
                turn['acts'].append(utt.act)
            else:
                if len(turn['ids']) > 0: 
                    turn['ids'] = [Utterance.tokenizer.cls_token_id] \
                    + turn['ids'] + [Utterance.tokenizer.sep_token_id]
                    turns.append(SimpleNamespace(**turn)) 
                turn = {'text':'', 'ids':[], 'acts':[], 'spkr':utt.spkr}
            prev_speaker = utt.spkr
        return turns
    
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

class Utterance:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    act_id_dict = load_json(f'{BASE_DIR}/swbd_act_dict.json')
    act_names_dict = load_json(f'{BASE_DIR}/swbd_act_names.json')
    
    punct = True
    actions = True
    
    def __init__(self, text, act, spkr):
        self.text = ' '.join(text)
        if not self.action: 
            self.text = re.sub("<<.*?>>", "", self.text).strip()
            self.text = re.sub("<.*?>", "", self.text).strip()
            
        if not self.punct: 
            self.text = re.sub(r'[^\w\s]', '', self.text).lower()

        self.act = self.act_id_dict[act]
        self.act_str = self.act_names_dict[act]
        self.spkr = spkr
        self.ids = self.tokenizer(self.text).input_ids
    
    @classmethod
    def set_punct(cls, value):
        cls.punct = value
    
    @classmethod
    def set_action(cls, value):
        cls.action = value
        
    def __repr__(self):
        return self.text
