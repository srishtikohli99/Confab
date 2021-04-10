from IntentWithoutEntity import Prediction as PredictionWOE
from IntentWithEntity import Prediction as PredictionWE
# from .IntentWithoutEntity import Preprocessing, LoadingData, DesignModel
# from .IntentWithEntity import Preprocessing, LoadingData, DesignModel
# from . import LoadingData, Classifier, Predict
# import os
# import json

pred_objWOE = PredictionWOE()
print(pred_objWOE.test())
pred_objWE = PredictionWE()
print(pred_objWE.test())