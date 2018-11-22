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

X,y = load_data([filename],root=args.root)
Xr,yr,fn = load_events(X[:maxf],y[:maxf],min_length=length_window)
yr = np.array(yr)
Xt,yt =transform_reads(Xr,yr,lenv=length_window)


ntwk = model(typem=typem)
ntwk.load_weights(weight_name)

Predicts = []
for xt in Xt:
    Predicts.append(np.mean(ntwk.predict(xt)))


Predicts = np.array(Predicts)
pylab.hist(Predicts,bins=40)
print("Percent of reads < 0.3",np.sum(Predicts<0.3)/np.sum(Predicts))
pylab.savefig("histo_B_T_%s"%(weight_name[:-5])+extra+".pdf")

csvf = pd.read_csv(filename)
wn = weight_name[:-5]
csvf[wn] = [1 for _ in range(len(csvf))]

assert len(Predicts) == len(fn)
for f,r in zip(fn,Predicts):
    if r > 0.3:
        b=1
    else:
        b=0
    fnshort = f.replace("/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/","")
    csvf.loc[csvf["filename"]==fnshort,wn]=b

csvf.to_csv(filename, index=False)
