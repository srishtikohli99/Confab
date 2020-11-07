from .train import Prediction

predict = Prediction()

def FlairPredict(phrase):
    sentence = predict.predict(phrase)
    text = sentence.to_dict('ner')["text"]
    n = len(text)
    text3 =""
    start =0
    text2=""
    for entity in sentence.to_dict('ner')["entities"]:
        text3 += str(entity["labels"]).split(" ")[0][1:] + " "
        text2 += text[start:entity['start_pos']] + "entity_"+str(entity["labels"]).split(" ")[0][1:]
        start = entity['end_pos']

    text3=text3[:-1]
    text2+=text[start:n]
    print(text2)
    print(text3)
    return (text2,text3)

