from bureau import classifier
import os
import json
from sklearn.metrics import accuracy_score

with open(os.path.join(os.getcwd(),"config.json")) as f:
    config = json.load(f)

if config["entityExtractor"] == "flair":
    from Morph.flairEntity import predict as pred_flair
else:
    from Morph.bertEntity import predict



if __name__ == '__main__':
    
    actual=[]
    predicted=[]
    train_file_path = os.path.join(os.getcwd(),"data/Validate")
    print("VALIDATION DIRECTORY PATH")
    print(train_file_path)
    for file in os.listdir(train_file_path):
        print(str(file))
        if str(file) == ".DS_Store":
            continue
        f = open(os.path.join(train_file_path,str(file)))
        dat = json.load(f)
        for key in dat:
            for utt in dat[key]:
                sen = ""
                entitiess = {}
                for text in utt["data"]:
                    sen += text["text"]
                    if "entity" in text:
                        entitiess[text["text"]] = text["entity"]
                predicted.append(classifier.intent_classify(sen, entitiess))
                actual.append(key)
                # print(sen)
                # print(entitiess)
            
    # print(len(predicted))
    # print(len(actual))
    print(accuracy_score(actual,predicted))
    
    
    





