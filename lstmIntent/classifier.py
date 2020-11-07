from .IntentWithoutEntity import Prediction as PredictionWOE
from .IntentWithEntity import Prediction as PredictionWE
from .IntentWithoutEntity import Preprocessing, LoadingData, DesignModel
from .IntentWithEntity import Preprocessing, LoadingData, DesignModel
from .EntityClassifier import LoadingData, Classifier, Predict
import os
import json

with open(os.path.join(os.getcwd(),"config.json")) as f:
    config = json.load(f)
embedding = config["embedding"]
confidence = config["confidence"]
fallback = config["fallback"]

pred_objWOE = PredictionWOE()
pred_objWE = PredictionWE()
pred_classifier = Predict()

def intent_classify(phrase, entities):
    final = {}
    intent = ""
    maximum = 0
    phrase_modified = phrase
    print(entities)

    if entities is None:
        
        WOE, r = pred_objWOE.predict(phrase, embedding)
        print(WOE[r])

    else:
        if len(entities) == 0:
            
            WOE, r1 = pred_objWOE.predict(phrase, embedding)
            WE, r2 = pred_objWE.predict(phrase, embedding)
            maximum = 0
            for key in WE:
                if WE[key] + WOE[key] > maximum:
                    maximum = WE[key] + WOE[key]
                    r=key
            if maximum >= confidence*2 :
                print(r)
            else:
                print(fallback)

        
        else:

            sent = ""

            for k, v in entities.items():
                sent+=v + " "
                phrase_modified = phrase_modified.replace(k, "entity"+v)

            WOE, r1 = pred_objWOE.predict(phrase, embedding)
            WE, r2 = pred_objWE.predict(phrase_modified, embedding)
            r3 = pred_classifier.predict(sent)
            if r1 == r2 and r1 == r3:
                print(r1)
            else:
                maximum = 0
                for key in WE:
                    if WE[key] + WOE[key] > maximum:
                        maximum = WE[key] + WOE[key]
                        r=key
                if maximum >= confidence*2 :
                    print(r)
                else:
                    print(fallback)
