from bertEntity import predict
from lstmIntent import classifier
from flairEntity import predict as pred_flair
import os
import json


def entity_extraction_bert(phrase):

    entities = predict.predictor(phrase)
    return entities

if __name__ == '__main__':

    with open(os.path.join(os.getcwd(),"config.json")) as f:
        config = json.load(f)
    

    while True:
        print("Hit me up, ask...")
        phrase = input()
        if phrase == '0':
            break
        if config["entity"]:
            if config["entityExtractor"] == "flair":
                entities = pred_flair.FlairPredict(phrase)
            else:
                entities = entity_extraction_bert(phrase)
        else:
            entities = None
        classifier.intent_classify(phrase, entities)





