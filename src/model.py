"""
/src/model.py: 
classes and functions to store the models and predictors
"""

import os
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from IPython.display import clear_output
from IPython.core.debugger import set_trace
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Dropout, Flatten, BatchNormalization
#from keras import optimizers

import numpy as np
from datetime import datetime
from misc_funcs import MFCC_DIR,MODEL_DIR,WEIGHT_DIR,WAV_DIR,DE2EN,NUM2EN,FULL_EM,load_mfcc_data,get_mfcc



class CNN_classif(nn.Module):
    """
    CNN classifier: inspired from "Emotion Recognition from Speech" (Kannan Venkataramanan,Haresh Rengaraj Rajamohan,2019)
    https://www.researchgate.net/publication/338138024_Emotion_Recognition_from_Speech

    Attributes
    ----------
    convblock[i] : nn.Sequential
        various convolutional blocks
    linblock : nn.Sequential
        output layer
    Methods
    -------
    forward()
        regular forward overriding
    """
    def __init__(self):
        super(CNN_classif,self).__init__()
        self.convblock1 = nn.Sequential(
                                nn.Conv2d(1,8,kernel_size=13),
                                nn.BatchNorm2d(8),
                                nn.ReLU())
        self.convblock2 = nn.Sequential(
                                nn.Conv2d(8,8,kernel_size=13),
                                nn.BatchNorm2d(8),
                                nn.Dropout(0.33),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(2,1)))
        self.convblock3 = nn.Sequential(
                                nn.Conv2d(8,8,kernel_size=13),
                                nn.BatchNorm2d(8),
                                nn.Dropout(0.33),
                                nn.ReLU())

        self.convblock4 = nn.Sequential(
                                nn.Conv2d(8,8,kernel_size=2),
                                nn.BatchNorm2d(8),
                                nn.Dropout(0.33),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=(2,1)))
        self.linblock = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(1456,64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64,7))
        # 144<-->896
    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.linblock(x)
        return x
    
MODEL = CNN_classif()

def load_most_recent(model):
    """
    loads the weights of the most recently computed network in the `models` directory into the attribute model
    Args:
        model(CNN_classif): model to load the weights in
        model_dir: path to the directory where the weights are stored
    """
    file_name = max([file  for root, dirs, files in os.walk(WEIGHT_DIR, topdown=False) for file in files])
    print("Loading {}".format(file_name))

    model.load_state_dict(torch.load(os.path.join(WEIGHT_DIR,file_name)))
    #necessary for systematic results with the dropout layers
    model.eval()

def train_model(model, X_train, y_train,X_test,y_test,nb_epochs=5,lr=1e-4,optimz = optim.Adam,callback=None,history = []):
    """
    trains the model passed in attribute
    Args:
        model(CNN_classif): model to train
        X_train(torch.Tensor):
        y_train(torch.Tensor):
        X_test(torch.Tensor)
        y_test(torch.Tensor)
        nb_epochs(int)
        lr(float)
        optimz(torch.optim)
        history(list): list of dict where we store the metrics for each epoch
        callback(fn): function taking (model,X_train,X_test,history,criterion) to call at each epoch
        model_dir: path to the directory where the weights are stored
    Returns:
        history(list): modified history
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optimz(model.parameters(), lr = lr)
    batch_size = 20
    print("Start training Model")
    for e in range(nb_epochs):
        print("Epoch:{}/{}".format(e,nb_epochs))
        for X_batch,y_batch in zip(X_train.split(batch_size),
                                y_train.split(batch_size)):
            output_batch = model(X_batch)
            loss = criterion(output_batch,y_batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if callback is not None:
            callback(model,X_train,X_test,history,criterion)
    return history

def train_model_noval(model, X_train, y_train,nb_epochs=5,lr=2e-4,optimz = optim.AdamW):
    """
    trains the model passed in attribute
    Args:
        model(CNN_classif): model to train
        X_train(torch.Tensor):
        y_train(torch.Tensor):
        nb_epochs(int)
        lr(float)
        optimz(torch.optim)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optimz(model.parameters(), lr = lr)
    batch_size = 20
    print("Start training Model")
    for e in range(nb_epochs):
        print("Epoch:{}/{}".format(e,nb_epochs))
        for X_batch,y_batch in zip(X_train.split(batch_size),
                                y_train.split(batch_size)):
            output_batch = model(X_batch)
            loss = criterion(output_batch,y_batch)
            model.zero_grad()
            loss.backward()
            optimizer.step()
    


def run_model(nb_epochs=5):
    """
    train and save the model 
    Args:
        nb_epochs(int):number of epochs to run the current model for
    """
    file_names,data_f,targets = load_mfcc_data(MFCC_DIR)
    model = CNN_classif()
    data_f = torch.Tensor(data_f)
    targets = torch.Tensor(targets).long()
    train_model_noval(model,data_f,targets.long(),nb_epochs)
    name = datetime.now().strftime("%m_%d_%H%M")
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR,name))


def compute_pred(file_name):
    """
    compute the prediction for the given file according to the most recently computed model 
    Args:
        filename(str): name of the file (ex: 03a01Fa )
    Returns:
        (dic): dictionary containing the true target and the predicted one
    """
    load_most_recent(MODEL)
    target0 = FULL_EM[DE2EN[file_name[5]]]
    #load the corresonding mfcc file
    with open(os.path.join(MFCC_DIR,file_name + ".pkl"),'rb') as f:
        sample,target = pickle.load(f)
    sample = torch.Tensor(sample).unsqueeze(0)
    predicted = MODEL(sample).max(1)[1].item()
    #convert label to actual emotion
    predicted = FULL_EM[NUM2EN[predicted]]
    target = FULL_EM[NUM2EN[target]]
    assert(target==target0)
    return {'true_label':target,
            'predicted':predicted}
    
def get_LSTM():
    model = Sequential()
    model.add(LSTM(10, input_shape=(414, 39)))
    # model.add(Dropout(0.5))
    model.add(Dense(7, activation="relu"))
    """,
    model.add(BatchNormalization(axis=-1)),
    model.add(Dense(16, activation="tanh"))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(7, activation="softmax"))
    """
    opt = optimizers.Adam(learning_rate=1e3)
    model.compile(
        loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"],
    )
    return model

def predict_conf(y_true,y_pred):
    """
    return the confusion matrix for torch Tensors
    """
    return confusion_matrix(y_true.numpy(),y_pred.numpy())

def predict_accuracy(y_true,y_pred):
    """
    return the confusion matrix for torch Tensors
    """
    accuracy = (y_pred == y_true).sum().item() / y_true.shape[0]
    return accuracy

def callback(model,X_train,X_test,history,criterion):
    """
    callback for train_model
    Args:
        model(CNN_classif):model getting trained
        X_train,X_test(torch.Tensor):
        history(list)
        criterion(nn.loss)
    """
    hist = {}
    limit = X_train.shape[0]//2
    output_train = model(X_train[:limit])
    _,pred_train = output_train.max(1)
    output_test = model(X_test)
    _,pred_test = output_test.max(1)
    hist['accuracy_score'] = predict_accuracy(y_train[:limit],pred_train)
    hist['valid_acc'] = predict_accuracy(y_test,pred_test)
    #hist['train_confusion'] = predict_conf(y_train,pred_train)
    #hist['valid_confusion'] = predict_conf(y_test,pred_test)
    hist['train_loss'] = criterion(output_train,y_train[:limit])
    hist['valid_loss'] = criterion(output_test,y_test)
    print("Train: {:.2f}% Validation: {:.2f}% ".format(hist['accuracy_score']*100,hist['valid_acc']*100))
    history.append(hist)
    with open("models/history.pkl",'wb') as f:
        pickle.dump(history,f)
    torch.save(model.state_dict(), "models/04_20_1424")
    
"""
model = CNN_classif()
file_names, sfs, data, targets = load_wav_data()
X_train,X_test,y_train,y_test = train_test_split(data,targets,test_size=0.2, random_state=14)
X_train,y_train = data_augment(X_train,y_train,rndom_noise=True,shift=True,pitch=True)
X_train = get_mfcc(X_train,sfs[0])
X_test = get_mfcc(X_test,sfs[0])
history = []
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train).type(torch.long)
X_test = torch.Tensor(X_test)
y_test = torch.Tensor(y_test).type(torch.long)
train_model(model,X_train,y_train,X_test,y_test,nb_epochs=5,lr=0.0004,optimz=optim.AdamW,callback=callback,history=history)
"""
if __name__ == '__main__':
    #run_model(data_f,targets,nb_epochs=1)
    print(smart_model('03a04Fd'))
