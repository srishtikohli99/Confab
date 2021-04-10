import os

TRAIN_FILE_PATH = os.path.join(os.getcwd(), "data")
VALIDATE_FILE_PATH = os.path.join(os.getcwd(), os.path.join("data", "Validate"))
SYNONYMS_FILE_PATH = os.path.join(os.getcwd(), "synonyms.json")
AUGMENTED_TEXT_EXTENSION = "##AUG##.json"