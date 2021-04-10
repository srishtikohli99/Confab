from bureau import classifier
import os
import json
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

with open(os.path.join(os.getcwd(),"config.json")) as f:
    config = json.load(f)

if config["entityExtractor"] == "flair":
    from Morph.flairEntity import predict as pred_flair
else:
    from Morph.bertEntity import predict

@app.get("/bot/{utterance}")
def bot_response(utterance: str):
    phrase = utterance
    if config["entity"]:
        if config["entityExtractor"] == "flair":
            entities = pred_flair.FlairPredict(phrase)
        else:
            entities = entity_extraction_bert(phrase)
    else:
        entities = None


    intent = classifier.intent_classify(phrase, entities)
    return {"intent": intent, "entities": entities}

def entity_extraction_bert(phrase):

    entities = predict.predictor(phrase)
    return entities
