import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

# from pandarallel import pandarallel
# pandarallel.initialize()

import spacy
import json
from spacy.training import Example
import random

import ahocorasick
from copy import deepcopy

def format_each_data(text, poi, street, nlp):
    entities_POI = []
    entities_STREET = []
    _text = deepcopy(text)
    
    if isinstance(poi, str):
        aho = _build_aho([poi])
        aho.make_automaton()
        latest_char_idx = 0
        
        for end, (_, word) in aho.iter(_text):
            start = end - len(word) + 1
            if start < latest_char_idx:
                continue
            entities_POI.append((start, end + 1, 'POI'))
            _text = _text.replace(word, " " * len(word))
            latest_char_idx = end + 1
    if isinstance(street, str):
        aho = _build_aho([street])
        aho.make_automaton()
        latest_char_idx = 0

        for end, (_, word) in aho.iter(_text):
            start = end - len(word) + 1
            if start < latest_char_idx:
                continue

            entities_STREET.append((start, end + 1, 'STREET'))
            latest_char_idx = end + 1
    street_example = None
    poi_example = None
    if len(entities_POI) > 0:
        poi_example = Example.from_dict(nlp.make_doc(text), {"entities": entities_POI})
    if len(entities_STREET) > 0:
        street_example = Example.from_dict(nlp.make_doc(text), {"entities": entities_POI})
    
    return poi_example, street_example


def train_model(train_data, num_iter, savename, nlp_model=None):
    print("data size: ", len(train_data))
    if nlp_model == None:
        nlp = spacy.blank('id')
        ner = nlp.create_pipe("ner")
        nlp.add_pipe("ner")
        ner.add_label("STREET")
        ner.add_label("POI")
    else:
        nlp = nlp_model
    optimizer = nlp.begin_training()
    


    for itn in range(40):
        random.shuffle(train_data)
        losses = {}
        total_loss = 0
        for batch in spacy.util.minibatch(train_data, size=2):
            examples = []
            for text, entities in batch:
                doc = nlp.make_doc(text)                
                examples.append(Example.from_dict(doc, entities))
            nlp.update(examples, sgd=optimizer, losses=losses, drop=0.3)
            total_loss += losses['ner']
        print(itn, " : ", total_loss/len(train_data))
            

    nlp.to_disk("./"+savename)




if __name__ == "__main__":  
   
    
    filter_df = pd.read_csv("./preprocess_data/filter_split.csv")
    filter_ids = filter_df['id'].to_list()


    with open("./preprocess_data/entitie.json", "r") as read_file:
        objects = json.load(read_file)
    TRAINING_DATA = []
    for data in objects:
        entities = []
        idx = data["id"]
        if idx in filter_ids:
            text = data["text"]
            object_list  = data["entitie"]

            for label_object in object_list :
                start_offset = label_object[0]
                end_offset = label_object[1]
                label = label_object[2]
                entities.append((start_offset, end_offset, label))
            
            spacy_entry = (text, {"entities": entities})
            TRAINING_DATA.append(spacy_entry)
    print(len(TRAINING_DATA))
    print(TRAINING_DATA[0][1]['entities'])


    TRAINING_DATA = [d for d in TRAINING_DATA if len(d[1]['entities']) > 0 ]
    # TRAINING_DATA = [d for d in TRAINING_DATA  for ]
    print(len(TRAINING_DATA))

    model_name = "./filter_model"
    nlp_model = spacy.load(model_name)
    train_model(TRAINING_DATA, 40, "filter_model_2", nlp_model)


