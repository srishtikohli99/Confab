import numpy as np

import joblib
import torch

from . import config
from . import dataset
from . import engine
from .model import EntityModel
from itertools import groupby


def bert_token_reconstruct(resultTags, bert_tokens):
    result = {}
    finalTags = ['']
    finalTokens = ['']
    for i in range(1, len(resultTags) - 1):

        try:
            if bert_tokens[i][0] == '#' and bert_tokens[i][1] == '#':
                finalTokens[-1] = finalTokens[-1] + bert_tokens[i][2:]
                continue
        except:
            pass

        if resultTags[i] == finalTags[-1] and resultTags != 'O':
            finalTokens[-1] = finalTokens[-1] + ' ' + bert_tokens[i]
        else:
            finalTokens.append(bert_tokens[i])
            finalTags.append(resultTags[i])

    for i in range(1, len(finalTokens)):
        if not (finalTags[i] == 'O' or finalTags[i] == ''):
            result[finalTokens[i]] = finalTags[i]

    return result


# if __name__ == "__main__":
def predictor(sentence):

    meta_data = joblib.load(config.META_PATH)
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))

    # sentence = """
    # Play all of me by rahul bajaj
    # """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    # print(sentence)
    # print(tokenized_sentence)
    # print((config.TOKENIZER.decode(tokenized_sentence)))
    #print(config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence))
    bert_tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence)

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        tags=[[0] * len(sentence)]
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(device)))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)

        resultTags = list(enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)])


        # print(
        #     #[i[0] for i in groupby(resultTags)]
        #     [v for i, v in enumerate(resultTags) if i == 0 or v != resultTags[i - 1] or v == 'O'][1:-1]
        # )

        return bert_token_reconstruct(resultTags, bert_tokens)
        # print(
        #     enc_pos.inverse_transform(
        #         pos.argmax(2).cpu().numpy().reshape(-1)
        #     )[:len(tokenized_sentence)]
        # )
