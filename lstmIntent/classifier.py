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

def intent_classify(phrase, entities):
    pred_objWOE = PredictionWOE()
    pred_objWE = PredictionWE()
    pred_classifier = Predict()
    final = {}
    intent = ""
    maximum = 0
    phrase_modified = phrase

    if entities is None:
        WOE, r = pred_objWOE.predict(phrase, embedding)
        for key in WOE:
            print(key)
            print(WOE[key])


    else:
        sent=""
        for k, v in entities.items():
            sent+=v + " "
            phrase_modified = phrase_modified.replace(k, "entity"+v)

        print(sent)
        print(pred_classifier.predict(sent))
        WOE, r = pred_objWOE.predict(phrase, embedding)
        WE, r = pred_objWE.predict(phrase_modified, embedding)
        for key in WE:
            print(key)
            print(WOE[key])
            print(WE[key])
            # final[key] = WE[key] + WOE[key]
            # if final[key] > maximum:
            #     intent = key
            #     maximum = final[key]

        print(intent)
        print(WOE[r])
    #print(WE[r])
    # print(maximum)