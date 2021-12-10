from transformers import BertTokenizerFast, RobertaTokenizerFast, BigBirdTokenizer, ReformerTokenizerFast, AlbertTokenizerFast
from transformers import BertConfig, BertModel, RobertaModel, ElectraModel, BigBirdModel, AlbertModel

def get_tokenizer(system):
    if system in ['bert', 'electra', 'rand']: 
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    elif system == 'bert_cased':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    elif system == 'roberta': 
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    elif system == 'big_bird': 
        tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
    elif system == 'albert':
        tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
    else: raise Exception
    return tokenizer

def get_transformer(system):
    if   system ==       'bert': transformer = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
    elif system ==    'electra': transformer = ElectraModel.from_pretrained('google/electra-base-discriminator', return_dict=True)
    elif system ==       'rand': transformer = BertModel(BertConfig(return_dict=True))
    elif system == 'bert_cased': transformer = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    elif system ==    'roberta': transformer = RobertaModel.from_pretrained('roberta-base', return_dict=True)
    elif system ==   'big_bird': transformer = BigBirdModel.from_pretrained('google/bigbird-roberta-base', return_dict=True)
    elif system ==     'albert': transformer = AlbertModel.from_pretrained('albert-base-v2', return_dict=True)
    else: raise Exception
    return transformer
