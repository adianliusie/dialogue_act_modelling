import json

def load_json(path):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

def flatten(x):
    return [tok for sent in x for tok in sent]