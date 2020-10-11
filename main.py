from bertEntity import predict
from lstmIntent import classifier


def entity_extraction_bert(phrase):

    entities = predict.predictor(phrase)
    print(entities)
    return entities

if __name__ == '__main__':

    print("Select Morph:")
    print("1 -> BERT \n")
    morph_option = int(input())

    print("Select Bureau:")
    print("2 -> LSTM \n")
    bureau_option = int(input())

    while True:
        print("Hit me up, ask...")
        phrase = input()
        if phrase == '0':
            break
        entities = entity_extraction_bert(phrase)
        classifier.intent_classify(phrase, entities)





    #print(predict.predictor("play all of me by john legend"))
    # morph = {1: 'bertEntity'}





