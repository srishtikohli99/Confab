import os

TRAIN_FILE_PATH = os.path.join(os.getcwd(), "data")
VALIDATE_FILE_PATH = os.path.join(os.getcwd(), os.path.join("data", "Validate"))
SYNONYMS_FILE_PATH = os.path.join(os.getcwd(), "synonyms.json")
AUGMENTED_TEXT_EXTENSION = "##AUG##.json"
CONFIG = os.path.join(os.getcwd(),"config.json")
BUREAU_MODELS = os.path.join(os.getcwd(), os.path.join("bureau", "models"))
MAXLENGTH = os.path.join(BUREAU_MODELS, "maxlength.pkl")
WOE_WORD2VEC = os.path.join(BUREAU_MODELS, "WOEWord2Vec.model")
WOE_PREPROCESS_OBJ = os.path.join(BUREAU_MODELS, "WOEpreprocess_obj.pkl")
WOE_ID2INTENT = os.path.join(BUREAU_MODELS, "WOEid2intent.pkl")
WE_WORD2VEC = os.path.join(BUREAU_MODELS, "WEWord2Vec.model")
WE_PREPROCESS_OBJ = os.path.join(BUREAU_MODELS, "WEpreprocess_obj.pkl")
WE_ID2INTENT = os.path.join(BUREAU_MODELS, "WEid2intent.pkl")
WE_SENTENCE_TRANSORMERS = os.path.join(BUREAU_MODELS, "WEst")
STATISTICAL_CLASSIFIER = os.path.join(BUREAU_MODELS, "statisticalClf.pkl")