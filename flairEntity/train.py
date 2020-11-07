import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from difflib import SequenceMatcher
import re
import pickle
import flair
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TokenEmbeddings, TransformerWordEmbeddings, ELMoEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.trainers import ModelTrainer

class LoadingData():
            
    def __init__(self):
        train_file_path = os.path.join(os.getcwd(),"data")
        validation_file_path = os.path.join(os.getcwd(),"data","Validate")
        category_id = 0
        self.cat_to_intent = {}
        self.intent_to_cat = {}

        for dirname, _, filenames in os.walk(train_file_path):
            
            for filename in filenames:
                if str(filename) == "Validate" or str(filename) == "SmallTalk":
                    continue
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json","")
                self.cat_to_intent[category_id] = intent_id
                self.intent_to_cat[intent_id] = category_id
                category_id+=1
        print(self.cat_to_intent)
        print(self.intent_to_cat)
        '''Training data'''
        training_data = list() 
        for dirname, _, filenames in os.walk(train_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json","")
                training_data+=self.make_data_for_intent_from_json(file_path,intent_id,self.intent_to_cat[intent_id])
        self.train_data_frame = pd.DataFrame(training_data, columns =['text', 'annotation'])   

        self.train_data_frame = self.train_data_frame.sample(frac = 1)



        '''Validation data'''
        validation_data = list()    
        for dirname, _, filenames in os.walk(validation_file_path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                intent_id = filename.replace(".json","")
                validation_data +=self.make_data_for_intent_from_json(file_path,intent_id,self.intent_to_cat[intent_id])                
        self.validation_data_frame = pd.DataFrame(validation_data, columns =['text', 'annotation'])

        self.validation_data_frame = self.validation_data_frame.sample(frac = 1)


    def make_data_for_intent_from_json(self,json_file,intent_id,cat):
        json_d = json.load(open(json_file))         
        
        json_dict = json_d[intent_id]

        sent_list = list()
        for i in json_dict:
            
            each_list = i['data']
            sent =""
            enties = []
            for i in each_list:
                sent = sent + i['text']+ " "
                if 'entity' in i.keys():
                    enties.append((i['text'],i['entity']))
            sent =sent[:-1]
            for i in range(3):
                sent = sent.replace("  "," ")
            sent_list.append((sent,enties))
#         print(sent_list)
        return sent_list


def matcher(string, pattern):
    '''
    Return the start and end index of any pattern present in the text.
    '''
    match_list = []
    pattern = pattern.strip()
    seqMatch = SequenceMatcher(None, string, pattern, autojunk=False)
    match = seqMatch.find_longest_match(0, len(string), 0, len(pattern))
    if (match.size == len(pattern)):
        start = match.a
        end = match.a + match.size
        match_tup = (start, end)
        string = string.replace(pattern, "X" * len(pattern), 1)
        match_list.append(match_tup)
        
    return match_list, string

def mark_sentence(s, match_list):
    '''
    Marks all the entities in the sentence as per the BIO scheme. 
    '''
    word_dict = {}
    for word in s.split():
        word_dict[word] = 'O'
        
    for start, end, e_type in match_list:
        temp_str = s[start:end]
        tmp_list = temp_str.split()
        if len(tmp_list) > 1:
            word_dict[tmp_list[0]] = 'B-' + e_type
            for w in tmp_list[1:]:
                word_dict[w] = 'I-' + e_type
        else:
            word_dict[temp_str] = 'B-' + e_type
    return word_dict

def clean(text):
    '''
    Just a helper fuction to add a space before the punctuations for better tokenization
    '''
    filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ".", ":", ";", "<", "=", ">", "?", "@", "[",
               "\\", "]", "_", "`", "{", "}", "~", "'"]
    for i in text:
        if i in filters:
            text = text.replace(i, " " + i)
            
    return text

def create_data(df, filepath):
    '''
    The function responsible for the creation of data in the said format.
    '''
    with open(filepath , 'w') as f:
        for text, annotation in zip(df.text, df.annotation):
            text = clean(text)
            text_ = text        
            match_list = []
            for i in annotation:
                a, text_ = matcher(text, i[0])
                if len(a)!=0:
                    match_list.append((a[0][0], a[0][1], i[1]))
            d = mark_sentence(text, match_list)

            for i in d.keys():
                f.writelines(i + ' ' + d[i] +'\n')
            f.writelines('\n')

class Prediction:

    def __init__(self):

        self.model = SequenceTagger.load(os.path.join(os.getcwd(), 'flairEntity/resources/best-model.pt'))

    def predict(self, phrase):

        phrase = Sentence(phrase)
        self.model.predict(phrase)
        return phrase

    

if __name__ == '__main__':

    with open(os.path.join(os.getcwd(),"config.json")) as f:
        config = json.load(f)

    embedding = config["flairEmbedding"]

    load_data_obj = LoadingData()
    create_data(load_data_obj.train_data_frame, os.path.join(os.getcwd(), "flairEntity/train.txt"))
    create_data(load_data_obj.validation_data_frame, os.path.join(os.getcwd(), "flairEntity/test.txt"))
    columns = {0 : 'text', 1 : 'ner'}
    # directory where the data resides
    data_folder = './'
    # initializing the corpus
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file = os.path.join(os.getcwd(), "flairEntity/train.txt"),
                                dev_file = os.path.join(os.getcwd(), "flairEntity/test.txt"))
    # tag to predict
    tag_type = 'ner'
    # make tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    if embedding == "flair":
        embedding_types : List[TokenEmbeddings] = [
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward')
            ]
    elif embedding == "roberta":
        embedding_types : List[TokenEmbeddings] = [
            TransformerWordEmbeddings("roberta-base", layers="all", use_scalar_mix=True)
            ]
    elif embedding == "bert":
        embedding_types : List[TokenEmbeddings] = [
            TransformerWordEmbeddings("bert-base-uncased")
            ]
    else:
        embedding_types : List[TokenEmbeddings] = [
            WordEmbeddings('glove')
            ]


    embeddings : StackedEmbeddings = StackedEmbeddings(
                                 embeddings=embedding_types)

    tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)
    print(tagger)
    trainer : ModelTrainer = ModelTrainer(tagger, corpus)
    trainer.train(os.path.join(os.getcwd(), 'flairEntity/resources/best-model.pt'),
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=50)
    

    

    
