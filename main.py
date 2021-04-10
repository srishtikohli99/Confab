from bureau import classifier
import os
import json

with open(os.path.join(os.getcwd(),"config.json")) as f:
    config = json.load(f)

if config["entityExtractor"] == "flair":
    from Morph.flairEntity import predict as pred_flair
else:
    from Morph.bertEntity import predict


def entity_extraction_bert(phrase):

    entities = predict.predictor(phrase)
    return entities

if __name__ == '__main__':
    
    while True:
        
        print("Hit me up, ask...")
        phrase = input()
        if phrase == '0':
            break
        if config["entity"]:
            if config["entityExtractor"] == "flair":
                entities = pred_flair.predict(phrase)#FlairPredict(phrase)
            else:
                entities = entity_extraction_bert(phrase)
        else:
            entities = None
        print("here1")
        classifier.intent_classify(phrase, entities)





