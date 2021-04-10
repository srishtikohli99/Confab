import numpy as np 
import pandas as pd 
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
if __name__ == '__main__':
    from constants import *
else:
    from .constants import *

class LoadingData():
        
    def __init__(self, test = False):
        if test:
            train_file_path = TRAIN_FILE_PATH
        else:
            train_file_path = VALIDATE_FILE_PATH
        self.data = {}
        self.intent=[]
        self.phrases = []
        for file in os.listdir(train_file_path):
            if not str(file).endswith(".json"):
                continue
            else:
                f = open(os.path.join(train_file_path,str(file)))
                dat = json.load(f)
                for key in dat:
                    self.data[key] = dat[key]
        self.data_helper()


    def data_helper(self):

        sent_list = list()
        intents = list()
        for key in self.data:
            json_dict = self.data[key]
            for i in json_dict:
                each_list = i['data']
                sent =""
                for i in each_list:
                    if 'entity' in i.keys():
                        sent = sent + i['entity']+ " "
                sent =sent[:-1]
                for i in range(3):
                    sent = sent.replace("  "," ")
                if len(sent) > 0:
                    sent_list.append(sent)
                    intents.append(key)
        self.phrases.extend(sent_list)
        self.intent.extend(intents)
        


class Classifier:

    def __init__(self, clf):


        self.clf = clf
        if clf == "svm":
            self.pipe = Pipeline([('bow',CountVectorizer()),('clf',LinearSVC(probability=True))])
        elif clf == "svcRBF":
            self.pipe = Pipeline([('bow',CountVectorizer()),('clf',SVC(kernel='rbf', C=0.01, probability=True))])
        elif clf =="decTree":
            self.pipe = Pipeline([('bow',CountVectorizer()),('clf',DecisionTreeClassifier())])
        else:
            self.pipe = Pipeline([('bow',CountVectorizer()),('clf',RandomForestClassifier(max_depth=2, random_state=0, n_estimators=100))])
        

    def train(self, data):

        print("Training Classifier")
        self.pipe.fit(data.phrases,data.intent)
        print("Training Complete")
        intent = self.pipe.predict(data.phrases)
        accuracy = accuracy_score(data.intent,intent)
        print("Training Score")
        print(accuracy)
        with open(STATISTICAL_CLASSIFIER, "wb") as f:
                pickle.dump(self.pipe, f)
    


class Predict:

    def __init__(self):

        with open(STATISTICAL_CLASSIFIER, "rb") as f:
            self.pipe = pickle.load(f)

    def predict(self, phrase):
        
        y = self.pipe.predict([phrase])
        return y
    
    def test(self):

        data = LoadingData(test = True)
        y = self.pipe.predict(data.phrases)
        accuracy = accuracy_score(data.intent,y)
        print("Testing Score")
        print(accuracy)
        

if __name__ == '__main__':
    
    with open(CONFIG) as f:
        config = json.load(f)

    clf = "randomforest"

    if "clf" in config.keys():
        clf = config["clf"]

    data = LoadingData(test = True)
    train = Classifier(clf)
    train.train(data)
    pred = Predict()
    pred.test()



