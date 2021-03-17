import joblib
import torch
import torch.nn as nn
import transformers
import config

import numpy as np
import pandas as pd

import sklearn
import os
import re
import json
from sklearn import preprocessing
from sklearn import model_selection

from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
import csv 
import math

class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL_PATH
        )
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
    
    def forward(
        self, 
        ids, 
        mask, 
        token_type_ids,
        target_tag
    ):
        o1, _ = self.bert(
            ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids
        )

        bo_tag = self.bert_drop_1(o1)

        tag = self.out_tag(bo_tag)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)

        loss = loss_tag 

        return tag,loss

    
class EntityDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }

class LoadingData():
            
    def __init__(self, smallTalk = False, test=False):

        if test:
            tp = config.test_file_path
        else:
            tp = config.train_file_path
        self.id2intent = {}
        self.intent2id = {}
        self.category_id=0
        print("TRAINIING DIRECTORY PATH")
        print(tp)
        data = {}
        for file in os.listdir(tp):
            if str(file).endswith(".json"):
                f = open(os.path.join(tp,str(file)))
                print(str(file))
                dat = json.load(f)
                for key in dat:
                    data[key] = dat[key]
                    self.intent2id[key] = self.category_id
                    self.category_id+=1
        self.data = data
        training_data=self.data_helper()
        self.train_data_frame = pd.DataFrame(training_data, columns =['query', 'intent','category'])   
        for key in self.intent2id:
            self.id2intent[self.intent2id[key]]=key

    def data_helper(self):

        sent_list = list()
        for key in self.data:
            json_dict = self.data[key]
            for i in json_dict:
                each_list = i['data']
                sent =""
                for i in each_list:
                    sent = sent + re.sub(r'[^\w\s]','',i["text"])+" "
                for i in range(3):
                    sent = sent.replace("  "," ")
                sent_list.append((sent,key,self.intent2id[key]))
        return sent_list

data_train = LoadingData()
d = data_train.data
sentences1 = []
entities1 = []
for key in d:
    for ele in d[key]:
        sentence = []
        entity = []
        for text in ele["data"]: 
            words = re.sub(r'[^\w\s]','',text["text"]).split()
            if 'entity' in text:
                l=len(words)
                if l == 1:
                    entity.append(text['entity'])
                    
                elif l == 2:
                    entity.append(text['entity'])
                    entity.append(text['entity'])
                    
                else:
                    entity.append(text['entity'])
                    for word in words[1:-1]:
                        entity.append(text['entity'])
                    entity.append(text['entity'])
                    
            else:
                
                for word in words:
                    entity.append('O')
            sentence.extend(words)
        sentences1.append(sentence)
        entities1.append(entity)
        
print(len(sentences1))
print(len(entities1))



    
fields = ['Sentence #', 'Word', 'Tag']   
rows = []
for i in range(len(sentences1)):
    
    for j in range(len(sentences1[i])):
        row = []
        row.append("Sentence: "+str(i+1))
        row.append(sentences1[i][j])
        row.append(entities1[i][j])
        rows.append(row)
        

filename = config.BENCH_PATH
    
with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile)       
    csvwriter.writerow(fields)   
    csvwriter.writerows(rows) 


sizes = []
for i in range(len(sentences1)):
    sizes.append(len(sentences1[i]))

max_len = np.max(np.array(sizes))
print(max_len)

data_test = LoadingData(test=True)
d = data_test.data
sentences1_test = []
entities1_test = []

for key in d:
    for ele in d[key]:
        sentence = []
        entity = []
        for text in ele["data"]:
            words = re.sub(r'[^\w\s]','',text["text"]).split()
            if 'entity' in text:
                l=len(words)
                if l == 1:
                    entity.append(text['entity'])
                    
                elif l == 2:
                    entity.append(text['entity'])
                    entity.append(text['entity'])
                    
                else:
                    entity.append(text['entity'])
                    for word in words[1:-1]:
                        entity.append(text['entity'])
                    entity.append(text['entity'])
                    
            else:
                
                for word in words:
                    entity.append('O')
            sentence.extend(words)
        sentences1_test.append(sentence)
        entities1_test.append(entity)
        
print(len(sentences1_test))
print(len(entities1_test))


fields = ['Sentence #', 'Word', 'Tag'] 
rows = []
for i in range(len(sentences1_test)):
    
    for j in range(len(sentences1_test[i])):
        row = []
        row.append("Sentence: "+str(i+1))
        row.append(sentences1_test[i][j])
        row.append(entities1_test[i][j])
        rows.append(row)
        

filename = config.BENCH_TEST_PATH


with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile)    
    csvwriter.writerow(fields)   
    csvwriter.writerows(rows) 

    def train_fn(data_loader, model, optimizer, device, scheduler):
        model.train()
        final_loss = 0
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            optimizer.zero_grad()
            _, loss = model(**data)
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)


    def eval_fn(data_loader, model, device):
        model.eval()
        final_loss = 0
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(device)
            _, loss = model(**data)
            final_loss += loss.item()
        return final_loss / len(data_loader)



    def loss_fn(output, target, mask, num_labels):
        lfn = nn.CrossEntropyLoss()
        active_loss = mask.view(-1) == 1
        active_logits = output.view(-1, num_labels)
        active_labels = torch.where(
            active_loss,
            target.view(-1),
            torch.tensor(lfn.ignore_index).type_as(target)
        )
        loss = lfn(active_logits, active_labels)
        return loss


   

if 0:

    def process_data(data_path, test=False, ):
        df = pd.read_csv(data_path, encoding="latin-1")
        df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

        enc_tag = preprocessing.LabelEncoder()
        df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

        sentences = df.groupby("Sentence #")["Word"].apply(list).values
        tag = df.groupby("Sentence #")["Tag"].apply(list).values
        return sentences, tag, enc_tag

    sentences, tag, enc_tag = process_data(BENCH_PATH)


    meta_data = {
        "enc_tag": enc_tag
    }

    joblib.dump(meta_data, config.META_PATH)
    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(
        sentences, 
        tag, 
        random_state=42, 
        test_size=0.1
    )

    train_dataset = EntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = EntityDataset(
        texts=test_sentences, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(
                    nd in n for nd in no_decay
                )
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(
                    nd in n for nd in no_decay
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(
        len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    )
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = train_fn(
            train_data_loader, 
            model, 
            optimizer, 
            device, 
            scheduler
        )
        test_loss = eval_fn(
            valid_data_loader,
            model,
            device
        )
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss

if 1:


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

    tags_predict = []
    cnt = 1
    meta_data = joblib.load(config.META_PATH)

    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH,map_location=torch.device(device)))
    model.to(device)

    for sentence in data_train.train_data_frame["query"]:
        tokenized_sentence = config.TOKENIZER.encode(sentence)

        bert_tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence)
        sentence = sentence.split()
        test_dataset = EntityDataset(
            texts=[sentence], 
            tags=[[0] * len(sentence)]
        )

        

        with torch.no_grad():
            data = test_dataset[0] 
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, _ = model(**data)
            tags_test = list(enc_tag.inverse_transform(
                    tag.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)])

        
        result = bert_token_reconstruct(tags_test,bert_tokens)
        tags_predict.append(result)
        if cnt%1000 == 0:
            print(cnt)
        cnt+=1
    print(len(data_train.train_data_frame["query"]))

    predicted = []
    for query in data_train.train_data_frame["query"]:
        p = []
        for i in range(len(query.split())):
            p.append('O')
        predicted.append(p)

    def find_sub_list(sl,l):
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return ind,ind+sll-1

    index = []

    for i in range(len(data_train.train_data_frame["query"])):
        s = data_train.train_data_frame["query"][i].lower().split()
        for key in tags_predict[i]:
            k= key.lower().split()
            try:
                start, end = find_sub_list(k,s)
                for j in range(start, end+1,1):
                    predicted[i][j] = tags_predict[i][key]
            except:
                
                index.append(i)

    a=[]
    p=[]

    for i in range(len(entities1)):
        for j in range(len(entities1[i])):
            a.append(entities1[i][j])
            

    for i in range(len(predicted)):
        for j in range(len(predicted[i])):
            p.append(predicted[i][j])


    print(sklearn.metrics.accuracy_score(a, p))

    tags_predict = []
    cnt = 1
    meta_data = joblib.load(config.META_PATH)
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device(device)))
    model.to(device)

    for sentence in data_test.train_data_frame["query"]:

        tokenized_sentence = config.TOKENIZER.encode(sentence)

        bert_tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence)
        sentence = sentence.split()

        test_dataset = EntityDataset(
            texts=[sentence], 
            tags=[[0] * len(sentence)]
        )

        

        with torch.no_grad():
            data = test_dataset[0]
            for k, v in data.items():
                data[k] = v.to(device).unsqueeze(0)
            tag, _ = model(**data)
            tags_test = list(enc_tag.inverse_transform(
                    tag.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)])
        result = bert_token_reconstruct(tags_test,bert_tokens)

        tags_predict.append(result)
        if cnt%100 == 0:
            print(cnt)
        cnt+=1

    # print(len(data_test.train_data_frame["query"]))
    predicted = []
    for query in data_test.train_data_frame["query"]:
        p = []
        for i in range(len(query.split())):
            p.append('O')
        predicted.append(p)

        index=[]
    for i in range(len(data_test.train_data_frame["query"])):
        s = data_test.train_data_frame["query"][i].lower().split()
        for key in tags_predict[i]:
            k= key.lower().split()
            try:
                start, end = find_sub_list(k,s)
                for j in range(start, end+1,1):
                    predicted[i][j] = tags_predict[i][key]
            except:
                
                index.append(i)

    a=[]
    p=[]

    for i in range(len(entities1_test)):
        for j in range(len(entities1_test[i])):
            a.append(entities1_test[i][j])
            

    for i in range(len(predicted)):
        for j in range(len(predicted[i])):
            p.append(predicted[i][j])

    print(sklearn.metrics.accuracy_score(a, p))


    labels = ['O', 'restaurant_name', 'restaurant_type', 'state', 'timeRange',
        'spatial_relation', 'poi', 'served_dish', 'party_size_number',
        'country', 'city', 'sort', 'cuisine', 'facility',
        'party_size_description', 'object_type', 'location_name',
        'movie_name', 'object_location_type', 'movie_type', 'object_name',
        'rating_value', 'best_rating', 'rating_unit', 'object_select',
        'object_part_of_series_type', 'geographic_poi',
        'condition_description', 'current_location',
        'condition_temperature', 'music_item', 'playlist', 'artist',
        'playlist_owner', 'entity_name', 'track', 'service', 'year',
        'album', 'genre']
    cm = confusion_matrix(a, p, labels)
    print(cm)
    report = sklearn.metrics.classification_report(a, p)
    print(report)
