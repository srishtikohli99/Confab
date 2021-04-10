import json
import os

syn_path = open(os.path.join(os.getcwd(),"synonyms.json"))
synonyms = json.load(syn_path)
print(synonyms)

train_file_path = os.path.join(os.getcwd(), "data")
files = []

for file in os.listdir(train_file_path):
    if str(file).endswith(".json") and not str(file).endswith("AUG.json"):
        files.append(str(file))
    
for file in files:

    newJS = {}
    print(str(file))
    name = str(file)[:-5]+"AUG.json"
    path = os.path.join(train_file_path, name)
    f = open(os.path.join(train_file_path, file))
    dat = json.load(f)

    for key in dat:

        newJS[key] = []
        for synonym in synonyms:
            print("Generating data for synonym {}".format(synonym))

            for word in synonyms[synonym]:

                for utterance in dat[key]:

                    flag=0
                    new_utt = []

                    for text in utterance["data"]:

                        if text["text"].find(synonym) != -1:
                            flag = 1
                            text["text"] = text["text"].replace(synonym,word)
                        new_utt.append(text)

                    if flag == 1:
                        newJS[key].append({"data":new_utt})

    print(path)
    if len(newJS[key])>0:
        with open(path, 'w') as fp:
            json.dump(newJS, fp) 
