

# if __name__ == '__main__':

from .IntentWithoutEntity import Prediction as PredictionWOE
from .IntentWithEntity import Prediction as PredictionWE
from .IntentWithoutEntity import Preprocessing, LoadingData, DesignModel
from .IntentWithEntity import Preprocessing, LoadingData, DesignModel


def intent_classify(phrase, entities):
    pred_objWOE = PredictionWOE()
    pred_objWE = PredictionWE()
    final = {}
    intent = ""
    maximum = 0
    phrase_modified = phrase
    for k, v in entities.items():
        phrase_modified = phrase_modified.replace(k, "ENTITY"+v)

    WOE, r = pred_objWOE.predict(phrase)
    WE, r = pred_objWE.predict(phrase_modified)
    for key in WE:
        final[key] = WE[key] + WOE[key]
        if final[key] > maximum:
            intent = key
            maximum = final[key]

    print(intent)
    print(WOE[r])
    print(WE[r])
    print(maximum)