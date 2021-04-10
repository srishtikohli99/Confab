import numpy as np 
import pandas as pd 
import json
import os
import en_core_web_sm
from tensorflow import keras
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Bidirectional, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalMaxPooling1D
import pickle
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
import gensim
import keras
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras import Model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input, Layer, GlobalMaxPooling1D, LSTM, Bidirectional, Concatenate
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.utils import CustomObjectScope
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    from constants import *
else:
    from .constants import *

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

class LoadingData():
            
    def __init__(self, test=False):
        train_file_path = TRAIN_FILE_PATH
        if test:
            train_file_path = VALIDATE_FILE_PATH
        self.id2intent = {}
        self.intent2id = {}
        self.category_id=0
        print("TRAINIING DIRECTORY PATH")
        print(train_file_path)
        data = {}
        for file in os.listdir(train_file_path):

            if not str(file).endswith(".json"):
                continue

            else:
                f = open(os.path.join(train_file_path,str(file)))
                dat = json.load(f)
                for key in dat:
                    if key in data:
                        data[key].extend(dat[key])
                    else:
                        data[key] = dat[key]
                        self.intent2id[key] = self.category_id
                        self.category_id+=1
        self.data = data
        training_data=self.data_helper()
        self.train_data_frame = pd.DataFrame(training_data, columns =['query', 'intent','category'])   
        for key in self.intent2id:
            self.id2intent[self.intent2id[key]]=key
        self.train_data_frame = self.train_data_frame.sample(frac = 1)

    def data_helper(self):

        sent_list = list()
        for key in self.data:
            json_dict = self.data[key]
            for i in json_dict:
                each_list = i['data']
                sent =""
                for i in each_list:
                    if 'entity' not in i.keys():
                        sent = sent + i['text']+ " "
                sent =sent[:-1]
                for i in range(3):
                    sent = sent.replace("  "," ")
                sent_list.append((sent,key,self.intent2id[key]))
        return sent_list


class Preprocessing():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.spacy_model = en_core_web_sm.load()
        self.tokenizer = None

    def createData(self, data, maxLength, embedding):

        with open(SYNONYMS_FILE_PATH, "rb") as f:
            synonyms = json.load(f)
        self.tokenizer = Tokenizer(num_words=None)
        with open(MAXLENGTH, "rb") as f:
                self.max_len = pickle.load(f)
        self.x_train = data.train_data_frame['query'].tolist()
        self.y_train = data.train_data_frame['category'].tolist()
        self.tokenizer.fit_on_texts(list(self.x_train))
        for key in synonyms:
            if key in self.tokenizer.word_index:
                for synonym in synonyms[key]:
                    if synonym not in self.tokenizer.word_index:
                        self.tokenizer.word_index[synonym] = self.tokenizer.word_index[key]

        
        if embedding != "custom":
            for i in range(len(self.x_train)):
                self.x_train[i] = text_to_word_sequence(self.x_train[i])
            self.y_train = to_categorical(self.y_train)

        if embedding == "custom":
            self.x_train = self.tokenizer.texts_to_sequences(self.x_train)
            self.x_train = pad_sequences(self.x_train, maxlen=self.max_len)
            
            self.y_train = to_categorical(self.y_train)
        self.word_index = self.tokenizer.word_index
        print("Setting Maximum Length to : ")
        print(self.max_len)
        
    def getSpacyEmbeddings(self,sentneces):
        sentences_vectors = list()
        for item in sentneces:
            query_vec = self.spacy_model(item) 
            sentences_vectors.append(query_vec.vector)
        return sentences_vectors
    
@tf.keras.utils.register_keras_serializable()
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


class DesignModel():
    def __init__(self,preprocess_obj):
        self.model = None
        self.x_train = preprocess_obj.x_train
        self.y_train = preprocess_obj.y_train

    def Word2VecEmbed(self, preprocess_obj):
        
        print("Training Word2Vec")
        Word2VecModel = Word2Vec(self.x_train,size=10*math.floor(math.log(len(preprocess_obj.word_index),10)),min_count=1)
        print("Trained")
        Word2VecModel.save(WOE_WORD2VEC)
        return Word2VecModel
        
    def simple_rnn(self,preprocess_obj,classes, embedding):
        
        self.model = Sequential()
        if embedding == "Word2Vec":
            model = self.Word2VecEmbed(preprocess_obj)
            for i in range(len(self.x_train)):
                words = self.x_train[i]
                wv = []
                for w in words:
                    if w not in model.wv.vocab.keys():
                        continue
                    wvec = model.wv[w]
                    wv.append(wvec)
                self.x_train[i] = np.array(wv)
            self.x_train = np.array(self.x_train)
            self.x_train = keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=preprocess_obj.max_len, dtype='float32', padding='post', truncating='post')

        if embedding == "custom":
            self.model.add(Embedding(len(preprocess_obj.word_index) + 1,10*math.floor(math.log(len(preprocess_obj.word_index),10)),input_length=preprocess_obj.max_len))
        
        self.model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        self.model.add(LSTM(64, dropout=0.25, recurrent_dropout=0.2, return_sequences=True))
        self.model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
    def model_train(self,batch_size,num_epoch, folds_):

        y = []
        folds = StratifiedKFold(n_splits=folds_, shuffle=True, random_state=256)
        for i in range(len(self.y_train)):
            for j in range(len(self.y_train[i])):
                if self.y_train[i][j] == 1:
                    y.append(j)
        print(len(y))
        y = np.array(y)
        
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.x_train,y)):
            strLog = "fold {}".format(fold_)
            print(strLog)
            name = "WOE" + str(fold_) + ".h5"
            filepath = os.path.join(os.getcwd(), os.path.join(BUREAU_MODELS,name))
            call_back = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,  save_weights_only=False, mode='auto')
            checkpoints = [call_back]
            print("Fitting to model")
            self.model.fit(self.x_train[trn_idx], self.y_train[trn_idx], batch_size=batch_size, epochs=num_epoch, validation_data = (self.x_train[val_idx], self.y_train[val_idx]), callbacks=checkpoints)
            print("Model Training complete.")
            self.model.save(filepath)

    def bidir_lstm(self,preprocess_obj,classes, embedding,  dropout, recurrent_dropout, lr):

        ## Embedding Layer
        sequence_input = Input(shape=(preprocess_obj.max_len,))
        # self.model.add(Embedding(,,input_length=preprocess_obj.max_len))
        embedded_sequences = Embedding(len(preprocess_obj.word_index) + 1, 10*math.floor(math.log(len(preprocess_obj.word_index),10)))(sequence_input)
        ## RNN Layer
        lstm = Bidirectional(LSTM(32, return_sequences = True, dropout=dropout, recurrent_dropout=recurrent_dropout))(embedded_sequences)
        # Getting our LSTM outputs
        (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(32, return_sequences=True, return_state=True))(lstm)

        ## Attention Layer
        att_out=attention()(lstm)
        outputs=Dense(classes,activation='softmax')(att_out)
        model_attn = Model(sequence_input, outputs)

        adam = optimizers.Adam(lr=lr)
        model_attn.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
        self.model = model_attn

    def lstm_train(self,batch_size,num_epoch, folds_):
        y = []
        folds = StratifiedKFold(n_splits=folds_, shuffle=True, random_state=256)
        for i in range(len(self.y_train)):
            for j in range(len(self.y_train[i])):
                if self.y_train[i][j] == 1:
                    y.append(j)
        print(len(y))
        y = np.array(y)
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(self.x_train,y)):
            strLog = "fold {}".format(fold_)
            print(strLog)
            name = "WOEAttn" + str(fold_) + ".h5"
            filepath = os.path.join(os.getcwd(), os.path.join(BUREAU_MODELS,name))
            call_back = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True,  save_weights_only=False, mode='auto')
            checkpoints = [call_back]
            print("Fitting to model")
            self.model.fit(self.x_train[trn_idx], self.y_train[trn_idx], batch_size=batch_size, epochs=num_epoch, validation_data = (self.x_train[val_idx], self.y_train[val_idx]), callbacks=checkpoints)
            print("Model Training complete.")
            self.model.save(filepath)



class Prediction():
    def __init__(self, folds):

        self.folds = folds
        self.Models = []
        with open(CONFIG) as f:
            config = json.load(f)
        if "embedding" in config.keys():
            self.embedding = config["embedding"]
        preprocess_obj = CustomUnpickler(open(WOE_PREPROCESS_OBJ, 'rb')).load()
        if config["model"] == "LSTM":
            for i in range(self.folds):
                filename = "WOE" + str(i)+ ".h5"
                model= keras.models.load_model(os.path.join(BUREAU_MODELS,filename))
                self.Models.append(model)
        else:
            
            for i in range(self.folds):
                with CustomObjectScope({'AttentionLayer': attention}):
                    filename = "WOEAttn" + str(i)+ ".h5"
                    model= keras.models.load_model(os.path.join(BUREAU_MODELS, filename))
                    self.Models.append(model)
        self.tokenizer = preprocess_obj.tokenizer
        self.max_len = preprocess_obj.max_len
        with open(WOE_ID2INTENT, "rb") as f3:
            self.id2intent = pickle.load(f3)


    def Word2VecPredict(self, query):

        model = gensim.models.Word2Vec.load(WOE_WORD2VEC)
        query = text_to_word_sequence(query)
        wv=[]
        for w in query:
            if w not in model.wv.vocab.keys():
                continue
            wvec = model.wv[w]
            wv.append(wvec)
        query = np.array(wv)
        query = keras.preprocessing.sequence.pad_sequences([query], maxlen=preprocess_obj.max_len, dtype='float32', padding='post', truncating='post')
        return query
        
    
    def predict(self,query, embedding, model = "Attention", test=False):
        

        if embedding!="custom" and model == "LSTM":
            query = self.Word2VecPredict(query)
            pred = self.model.predict(query)
            resulti = {}
            result = self.id2intent[predi]
        
        elif embedding == "custom" and model == "LSTM":
            query_seq = self.tokenizer.texts_to_sequences([query])
            query_pad = pad_sequences(query_seq, maxlen=self.max_len)
            for i in range(self.folds):
                pred = self.Models[i].predict_step(query_pad)
                # print(pred)
                if i == 0:
                    p = np.zeros(len(pred[0]))
                    p=[p]
                    # print(p)
                for j in range(len(pred[0])):
                    p[0][j] += pred[0][j]/self.folds
            pred = p


        elif model == "Attention":
            
            query_seq = self.tokenizer.texts_to_sequences([query])
            query_pad = pad_sequences(query_seq, maxlen=self.max_len)
            for i in range(self.folds):
                pred = self.Models[i].predict_step(query_pad)
                # print(pred)
                if i == 0:
                    p = np.zeros(len(pred[0]))
                    p=[p]
                    # print(p)
                for j in range(len(pred[0])):
                    p[0][j] += pred[0][j]/self.folds
            pred = p
            # print(pred)
            # i=0
            # pred = self.Models[i].predict(query_pad)
            # # print(pred)
            # if i == 0:
            #     p = np.zeros(len(pred[0]))
            #     p=[p]
            #     # print(p)
            # for j in range(len(pred[0])):
            #     p[0][j] += pred[0][j]/self.folds
            # pred = p
            # # print(pred)

        predi = np.argmax(pred)
        if test:
            return predi
        result = self.id2intent[predi]
        # print("WithEntity" + result)
        resulti = {}
        for i in range(len(pred[0])):
            resulti[self.id2intent[i]] = pred[0][i]
        return resulti, result

    def test(self):
        data = LoadingData(test=True)
        samples = data.train_data_frame['query'].tolist()
        labels = data.train_data_frame['category'].tolist()
        predictedLabels = []
        for sample in samples:
            i = self.predict(sample, self.embedding, test=True)
            predictedLabels.append(i)
        score = accuracy_score(np.array(predictedLabels),np.array(labels))
        return score
        

if __name__ == '__main__':

    with open(CONFIG) as f:
        config = json.load(f)
    epochs = 20
    batchSize = 64
    maxLength = 0
    embedding = "custom"
    if "epochs" in config.keys():
        epochs = config["epochs"]
    if "batchSize" in config.keys():
        batchSize = config["batchSize"]
    if "maxLength" in config.keys():
        maxLength = config["maxLength"]
    if "embedding" in config.keys():
        embedding = config["embedding"]
    if "model" in config.keys():
        model = config["model"]
    if "folds" in config.keys():
        folds = config["folds"]
    if "dropout" in config.keys():
        dropout = config["dropout"]
    if "recurrent_dropout" in config.keys():
        recurrent_dropout = config["recurrent_dropout"]
    if "lr" in config.keys():
        lr = config["lr"]
    
    data = LoadingData()
    preprocess_obj = Preprocessing()
    preprocess_obj.createData(data, maxLength, embedding)
    model_obj = DesignModel(preprocess_obj)
    if model == "LSTM":
        model_obj.simple_rnn(preprocess_obj,data.category_id, embedding)
        model_obj.model_train(batchSize,epochs, folds)
    elif model == "Attention":
        model_obj.bidir_lstm(preprocess_obj,data.category_id, embedding)
        model_obj.lstm_train(batchSize,epochs, folds)

    with open(WOE_PREPROCESS_OBJ,"wb") as f:
        pickle.dump(preprocess_obj,f)

    with open(WOE_ID2INTENT,"wb") as f3:
        pickle.dump(data.id2intent,f3)

