import os
import json

train_file_path = os.path.join(os.getcwd(), "data/Validate")
print("Training Data Directory")
print(train_file_path)
for file in os.listdir(train_file_path):

    if str(file) == "Validate":
        continue
    if str(file) == ".DS_Store":
        continue

    if str(file) != "SmallTalk":
        path = os.path.join(train_file_path, str(file))
        print(path)
        f = open(path)
        dat = json.load(f)
        for key in dat:
            print(len(dat[key]))