from types import SimpleNamespace
import json
from tqdm import tqdm 
import string
import re 
import time

from src.utils import load_json, get_tokenizer, flatten, load_list

class ConvHandler:
    """
    class for conveniently dealing with conversations
    """
    cache = {}
    
    def __init__(self, data_src:str, system:str='bert', class_reduct:bool=False, 
                 punct:bool=True, action:bool=True, pre_proc=False, lim:int=None): 
        """
        init to convert all json conversations to a Conversation object 

        Args:
            data_src: data set to be used
            system:   tokenizer/model to be used for experiments
            punc:     whether punctuation should be used
            action:   whether actions should be used
            lim:      conv limit for data (used for debugging)
            class_reduct:  whether to use reduced classes
        """
        
        TextCleaner.set_options(punct=punct, action=action, pre_proc=pre_proc)
        Utterance.set_system(system)
        self.tokenizer = Utterance.tokenizer
        
        #use hashing so that if data processed recently, use the processed version
        arg_hash = hash((data_src, system, punct, action, class_reduct, lim))
        if arg_hash in self.cache:
            self.__dict__ = self.cache[arg_hash]
        else:
            self.get_data(data_src, lim, class_reduct)   
            self.__class__.cache[arg_hash] = self.__dict__

    def get_data(self, data_src:str, lim:int=None, class_reduct:bool=False):
        """routing to function to load the right data set"""
        
        if data_src in ['swbd', 'ami']: 
            self.get_data_sets(data_src, lim, class_reduct)
        elif data_src in ['oet', 'swbd_asr', 'oet_asr']: 
            self.get_trans_data(data_src)
        else:
            self.get_trans_from_path(data_src)
            
    def get_data_sets(self, data_src:str, lim:int=None, class_reduct:bool=False):
        base_dir  = f'/home/alta/Conversational/OET/al826/2021/data/{data_src}'
        self.data = load_json(f'{base_dir}/data.json')
        
        self.train_ids = load_list(f'{base_dir}/id_splits/ASR_train_ids') 
        self.dev_ids   = load_list(f'{base_dir}/id_splits/ASR_dev_ids') 
        self.test_ids  = load_list(f'{base_dir}/id_splits/ASR_test_ids') 
        
        train = [i for i in self.data if i['conv_id'] in self.train_ids]
        dev   = [i for i in self.data if i['conv_id'] in self.dev_ids]
        test  = [i for i in self.data if i['conv_id'] in self.test_ids]       
                
        if class_reduct:
            Utterance.load_label_dict(f'{base_dir}/red_labels.json')
            Utterance.reduced_dict(f'{base_dir}/red_id_map.json')
        else:
            Utterance.load_label_dict(f'{base_dir}/labels.json')

        self.label_dict = Utterance.label_dict
        
        if lim:  train, dev, test = train[:lim], dev[:lim], test[:lim]
            
        self.train = [Conversation(conv) for conv in tqdm(train)]
        self.dev   = [Conversation(conv) for conv in tqdm(dev)] 
        self.test  = [Conversation(conv) for conv in tqdm(test)] 
    
    def get_trans_data(self, name, lim:int=None):
        base_dir = f'/home/alta/Conversational/OET/al826/2021/data/{name}'
        data = load_json(f'{base_dir}/data.json')
        if lim:  data = data[:lim]
        self.test = [Conversation(conv) for conv in tqdm(data)]
        self.test = [conv for conv in self.test if max([len(turn.ids) for turn in conv.turns])<500]
        
    def get_trans_from_path(self, path, lim:int=None):
        """ for the cases where conversation not saved in usual location"""
        
        data = load_json(path)
        if lim:  data = data[:lim]
        self.test = [Conversation(conv) for conv in tqdm(data)]

    def __getitem__(self, x:str):
        x = str(x)
        for dset in [getattr(self, x) for x in ['train', 'dev', 'test'] if hasattr(self, x)]:
            for conv in dset:
                if conv.conv_id == x: return conv
        raise ValueError('conversation not found')
             
    def __contains__(self, x:str):
        output = False
        for dset in [getattr(self, x) for x in ['train', 'dev', 'test'] if hasattr(self, x)]:
            if x in [conv.conv_id for conv in dset]:
                output = True
        return output
    
class Conversation:
    def __init__(self, data:dict):
        self.data = data
        self.utts  = [Utterance(**utt) for utt in self.data['turns']]
        self.turns = self.make_turns()
        self.lab_seq = [utt.label for utt in self.utts]
        
        for key, value in data.items():
            if key != 'turns':
                setattr(self, key, value)
        
        self.conv_id = str(self.data['conv_id']) #done above but here for doc
        
    def make_turns(self):
        """ 'makes' turns by joining utts from the same speaker"""
        
        prev_speaker = None
        turn = {'text':'', 'speaker':None, 'tags':{'segs':[], 'labels':[]}}
        turns, i = [], 0

        for utt in self.utts:
            #if speaker changes, turn ends
            if utt.speaker != prev_speaker:                    
                if len(turn['text'])>0:
                    #Make turn obj and add to list of turns
                    turn_obj = Utterance(**turn)
                    if len(turn['text'])>0:                    
                        turns.append(turn_obj) 
                        assert turn_obj.ids[1:-1] == flatten(turn['tags']['ids'])

                turn = {'text':'', 'speaker':utt.speaker, 
                        'tags':{'segs':[], 'labels':[], 'ids':[]}}
            
            #add information of current utterance to turn
            turn['text'] += ' ' + utt.text
            turn['tags']['labels'].append(utt.label)
            turn['tags']['ids'].append(utt.ids[1:-1])
            turn['tags']['segs'].append(len(flatten(turn['tags']['ids']))+1)
            prev_speaker = utt.speaker
            
        #The last utterance shouldn't be ignored
        if len(turn['text'])>0:
            turn_obj = Utterance(**turn)
            turns.append(turn_obj) 
        return turns        
        
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

    def __add__(self, other):
        output_data = {}
        for key in self.data.keys():
            output_data[key] = self.data[key] + other.data[key]
        return Conversation(output_data)
        
class Utterance:
    reduced, tokenizer = False, None

    def __init__(self, text, speaker=None, label=None, tags=None):
        self.og_text = text
        self.text = TextCleaner.clean_text(self.og_text)
        self.speaker = speaker
        self.label = label
        self.tags = tags
        if self.tokenizer: self.ids = self.tokenizer(self.text).input_ids
        else:              self.ids = [None, None]
            
        if label != None:
            if self.reduced:
                self.label = self.id_map[label]
            self.label_name = self.label_dict[self.label]
            
    @classmethod
    def set_system(cls, system:str):
        if system:
            cls.tokenizer = get_tokenizer(system)

    @classmethod
    def load_label_dict(cls, path):
        label_dict = load_json(path)
        cls.label_dict = {int(k):v for k, v in label_dict.items()}
        
    @classmethod
    def reduced_dict(cls, path):
        cls.reduced = True
        id_map = load_json(path)
        cls.id_map = {int(k):v for k, v in id_map.items()}

    def __repr__(self):
        return self.text

class TextCleaner:
    punct, action, pre_proc = True, True, False

    @classmethod
    def clean_text(cls, text:str)->str:
        """method which cleans text with chosen convention"""
        
        if cls.pre_proc:
            text = cls.expand_contract(text)
            text = cls.acronyms(text)
        if not cls.action:
            text = re.sub("[\[\(\<\%].*?[\]\)\>\%]", "", text)    
        if not cls.punct: 
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
        
        text = cls.hesitation(text)
        #text = cls.repetition(text)
        text = text.strip()
        return text

    @staticmethod
    def expand_contract(text):
        """function which expands contractions"""
        contractions = {"'ve":" have", "'m":" am", "'ll":" will", "'d":" would",
                        "'re":" are", "'s":" is"}
        for c, c_exp in contractions.items():
            if c in text:
                text = text.replace(c, c_exp)
        return text

    @staticmethod
    def acronyms(text):
        """expands acronyms"""
        return text.replace('_', ' ')

    @staticmethod
    def hesitation(text):
        """internal function which converts hesitation to standard ones"""
        hes_maps = {"umhum":"uhhuh", "uh-huh":"uhhuh","hum":"um", "uh":'um'}
        #hes_maps = {"umhum":"", "uh-huh":"","hum":"", "uh":'', 'uhhuh':'', 'um':''}

        for h1, h2 in hes_maps.items():
            if h1 in text:
                pattern = r'(^|[^a-zA-z])'+h1+r'($|[^a-zA-Z])'
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
                text = re.sub(pattern, r'\1'+h2+r'\2', text) #twice as uh uh share middle character
        return text 

    @staticmethod
    def repetition(text):
        """removes all words that end in -, i.e. (is tha- that a) -> (is that a)"""
        text = re.sub(r'\b\w+-\s', '', text)
        return text
    
    @classmethod
    def set_options(cls, punct=True, action=True, pre_proc=False):
        cls.punct = punct
        cls.action = action
        cls.pre_proc = pre_proc
    