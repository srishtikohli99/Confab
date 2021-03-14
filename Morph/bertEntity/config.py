import transformers
import os
BASE_PATH = os.path.join(os.getcwd(), 'Morph/bertEntity/')
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 5
BASE_MODEL_PATH = BASE_PATH + "models/bert_base_uncased"
MODEL_PATH = BASE_PATH + "models/model.bin"
META_PATH = BASE_PATH + "models/meta.bin"
TRAINING_FILE = "input/dataset.csv"
JSON_FILE = "input/JsonDataFiles/"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case = True
)
