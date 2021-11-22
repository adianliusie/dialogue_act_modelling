import json

def load_json(path):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data
