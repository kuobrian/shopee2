import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', None)

# from pandarallel import pandarallel
# pandarallel.initialize()

import spacy
from spacy.training import Example
import random

import ahocorasick
from copy import deepcopy
import json
import spacy
import random

def _build_aho(words):
    aho = ahocorasick.Automaton()
    for idx, key in enumerate(words):
        
        aho.add_word(key, (idx, key))

    return aho

def format_data(text, poi, street):
    entities = []
    _text = deepcopy(text)
    
    if isinstance(poi, str):
        aho = _build_aho([poi])
        aho.make_automaton()
        latest_char_idx = 0
        
        for end, (_, word) in aho.iter(_text):
            start = end - len(word) + 1
            
            if start < latest_char_idx:
                continue

            entities.append([start, end + 1, 'POI'])
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

            entities.append([start, end + 1, 'STREET'])
            latest_char_idx = end + 1
    
    return text, entities


def extract_entities(row):
        extracted = row['POI/street'].split("/")
        
        if len(extracted) == 2:
            poi, street = extracted
            if poi.strip() != '':
                row['POI'] = poi
            
            if street.strip() != '':
                row['street'] = street
            
        return row


if __name__ == "__main__":
    
    
    
    # df = pd.read_csv("./Address Elements Extraction Dataset/train.csv")
    # df.set_index("id", inplace=True)
    # df['POI'] = np.nan
    # df['street'] = np.nan
    # df = df.apply(extract_entities, axis=1)
    # nlp = spacy.blank('id')  # create blank Language class
    # print("Preparing Spacy examples...")

    # examples = []
    
    # for idx in df.index:
    #     try:
    #         row = df.loc[idx]
    #         text, entitie = format_data(row['raw_address'], row['POI'], row['street'])

            
    #         examples.append({"id": idx, "text": text, "entitie": entitie})
    #     except Exception as e:
    #         print(idx)
    #         print("-" * 50)
    #         print(e)
    #         break

    # with open('./entitie.json','w') as f:
    #     # f.write(json.dumps(examples,  indent=2))
    #     json.dump(examples, f, indent=2)


    TRAINING_DATA = []

    with open("./entitie.json", "r") as read_file:
        objects = json.load(read_file)
    
        
    for data in objects:
        entities = []
        # id = data["id"]
        text = data["text"]
        object_list  = data["entitie"]

        for label_object in object_list :
            start_offset = label_object[0]
            end_offset = label_object[1]
            label = label_object[2]
            entities.append((start_offset, end_offset, label))
        
        spacy_entry = (text, {"entities": entities})
        TRAINING_DATA.append(spacy_entry)


    spacy.prefer_gpu()
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner")
    ner.add_label("STREET")
    ner.add_label("POI")

    nlp.begin_training()
    # Loop for 40 iterations
    for itn in range(40):
        random.shuffle(TRAINING_DATA)
        losses = {}
        examples = []
        for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
            for text, entities in batch:
                # texts = [text for text, entities in batch]
                # annotations = [entities for text, entities in batch]
                doc = nlp.make_doc(text)
                # spacy.training.offsets_to_biluo_tags(doc, entities)
                example = Example.from_dict(doc, entities)
                examples.append(example)
            nlp.update(examples, losses=losses, drop=0.3)
            print(itn, " : ", losses)
            example = []

    nlp.to_disk("./address_model")


