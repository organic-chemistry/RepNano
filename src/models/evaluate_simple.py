from .simple_utilities import load_data,load_events,transform_reads
from keras.models import Sequential
from keras.layers import Dense,Flatten,TimeDistributed,AveragePooling1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import pylab
import pandas as pd

def model(typem=1,window_length=None):
    if typem==1:


        if window_length is None:
            lenv=200
        else:
            lenv=window_length

        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(lenv, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
        ntwk=model


    if typem==2:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu', input_shape=(256, 1)))
        model.add(MaxPooling1D(pool_size=4)) # 64
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4)) #16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))

        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        ntwk=model
        lenv=256

    if typem==3:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu', input_shape=(256, 1)))
        model.add(MaxPooling1D(pool_size=4)) # 64
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4)) #16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                                 activation='relu'))

        model.add(TimeDistributed(Dense(1, activation='sigmoid')))

        model.add(AveragePooling1D(pool_size=16))
        model.add(Flatten())
        ntwk =model
        lenv=256
    return ntwk,lenv



import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--file', dest='filename', type=str)
parser.add_argument('--extra', dest='extra', type=str,default="")
parser.add_argument('--root', dest='root', type=str,default="/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/")
parser.add_argument('--weight-name', dest='weight_name', type=str)
parser.add_argument('--typem', dest='typem', type=int,default=1)
parser.add_argument('--maxf', dest='maxf', type=int,default=None)
parser.add_argument('--window-length', dest='length_window', type=int,default=None)
parser.add_argument('--overlap', dest='overlap', type=int,default=None)
parser.add_argument('--delta', dest="delta", action="store_true")



args = parser.parse_args()

filename=args.filename
length_window=args.length_window
maxf=args.maxf
weight_name=args.weight_name
typem=args.typem
extra=args.extra

ntwk,lenv = model(typem=typem,window_length=length_window)
ntwk.load_weights(weight_name)

if length_window is None:
    length_window = lenv

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
    X,y = load_data([t],root=args.root,values=["test_longueur_lstm_from_scratch_without_human_weights.25-0.02","init_B"])
    Xrt,yrt,fnt = load_events(X[:args.maxf],y[:args.maxf],min_length=2*length_window)
    Xt,yt =transform_reads(Xrt,np.array(yrt),lenv=length_window,max_len=2000,overlap=args.overlap,delta=args.delta)

    data[t.split("/")[-1][:-4]]=[Xt,[yti[0][0] for yti in yt]]




Predicts = []

closer = ["T-yeast","T1-yeast","B-9-yeast","B-40-yeast","B-69-yeast","B-yeast","B1-yeast"]
Xr=[]
yr=[]
for d in closer:
    print(d)
    yr.extend(data[d][1])
    for xt in data[d][0]:
        if args.overlap is None:
            Predicts.append(np.mean(ntwk.predict(xt)))
        else:
            #print(xt)
            xt=np.array(xt)
            #print(xt.shape)
            r = ntwk.predict(xt.reshape(-1,length_window,xt.shape[-1]))
            #print(len(r))
            r= r.reshape(args.overlap,-1,1)
            #print(r.shape)
            Predicts.append(np.mean(np.median(r,axis=0)))


Predicts2 = []
closer = ["B-9-yeast","B-yeast"]
Xr=[]
yr2=[]
for d in closer:
    print(d)
    yr2.extend(data[d][1])
    for xt in data[d][0]:
        if args.overlap is None:
            Predicts2.append(np.mean(ntwk.predict(xt)))
        else:
            #print(xt)
            xt=np.array(xt)
            r = ntwk.predict(xt.reshape(-1,length_window,xt.shape[-1])).reshape(args.overlap,-1,1)
            Predicts2.append(np.mean(np.median(r,axis=0)))

Predicts = np.array(Predicts)
Predicts2 = np.array(Predicts2)

pylab.title("Deviation %.2f, on test only (9 and B1) %.2f" % (np.std(Predicts-yr),np.std(Predicts2-yr2)))
pylab.plot(Predicts)
pylab.plot(yr,"o")
pylab.xlabel("Sample #")
pylab.ylabel("Ratio_b")
filen =weight_name[:-5]+"sample_values"+extra+".pdf"
print("Writing on %s"% filen)
pylab.savefig(filen)
