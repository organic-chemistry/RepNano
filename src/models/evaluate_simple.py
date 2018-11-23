from .simple_utilities import load_data,load_events,transform_reads
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import pylab
import pandas as pd

def model(typem=1):
    if typem==1:

        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(200, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
        ntwk=model
    return ntwk



import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', dest='filename', type=str)
parser.add_argument('--extra', dest='extra', type=str,default="")
parser.add_argument('--root', dest='root', type=str,default="/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/")
parser.add_argument('--weight-name', dest='weight_name', type=str)
parser.add_argument('--typem', dest='typem', type=int,default=1)
parser.add_argument('--maxf', dest='maxf', type=int,default=None)
parser.add_argument('--window-length', dest='length_window', type=int,default=200)


args = parser.parse_args()

filename=args.filename
length_window=args.length_window
maxf=args.maxf
weight_name=args.weight_name
typem=args.typem
extra=args.extra

train_test=['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-yeast.csv',
            '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv',
            '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T1-yeast.csv',
            '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-69-yeast.csv',
            '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv',
            '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-40-yeast.csv',
            '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B1-yeast.csv']
data = {}
for t in train_test:
    print(t)
    ws=5
    if "T-yeast" in t:
        #print("la")
        ws=8
    #print(ws)
    X,y = load_data([t],root=args.root)
    Xrt,yrt,fnt = load_events(X[:40],y[:40],min_length=length_window)
    Xt,yt =transform_reads(Xrt,np.array(yrt),lenv=length_window,max_len=2000)

    data[t.split("/")[-1][:-4]]=[Xt,[yti[0][0] for yti in yt]]


ntwk = model(typem=typem)
ntwk.load_weights(weight_name)

Predicts = []

closer = ["T-yeast","T1-yeast","B-9-yeast","B-40-yeast","B-69-yeast","B-yeast","B1-yeast"]
Xr=[]
yr=[]
for d in closer:
    print(d)
    yr.extend(data[d][1])
    for xt in data[d][0]:
        Predicts.append(np.mean(ntwk.predict(xt)))


Predicts = np.array(Predicts)
pylab.plot(Predicts)
pylab.plot(yr)
pylab.xlabel("Sample #")
pylab.ylabel("Ratio_b")
pylab.savefig("sample_values_%s"%(weight_name[:-5])+extra+".pdf")
