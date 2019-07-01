import argparse
from repnano.models.simple_utilities import load_data, load_events, transform_reads
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, AveragePooling1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from repnano.models.create_model import create_model
import numpy as np
import pylab
import pandas as pd
import os


def model(typem=1, base=False, nc=1, window_length=None):
    init = 1
    if base:
        init = 5

    if typem == 1:

        if window_length is None:
            lenv = 200
        else:
            lenv = window_length

        """
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length,
        input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                         activation='relu', input_shape=(lenv, init)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
        """
        params = {"filters": 32, "kernel_size": 3,
                  "choice_pooling": {"pooling": True, "pool_size": 2},
                  "neurones": 100, "batch_size": 50, "optimizer": "adam",
                  "activation": "sigmoid", "nc": nc, "dropout": 0,"bi":False}
        ntwk = create_model(params, create_only=True)
    if typem == 7:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu', input_shape=(96, init)))
        """
        model.add(MaxPooling1D(pool_size=4)) # 16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4)) #4
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                                 activation='relu'))

        #model.add(LSTM(100))
        #model.add(Dense(1, activation='linear'))
        """
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(AveragePooling1D(pool_size=6))
        model.add(Flatten())
        ntwk = model
        lenv = 96
    ntwk.summary()
    return ntwk


parser = argparse.ArgumentParser()

parser.add_argument('--file', dest='filename', type=str)
parser.add_argument('--extra', dest='extra', type=str, default="")
parser.add_argument('--root', dest='root', type=str,
                    default="/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/")
parser.add_argument('--weight-name', dest='weight_name', type=str)
parser.add_argument('--typem', dest='typem', type=int, default=1)
parser.add_argument('--nc', dest='nc', type=int, default=1)
parser.add_argument('--target', dest='target', type=int, default=1)
parser.add_argument('--thres', dest='thres', type=float, default=0.3)
parser.add_argument('--assign-value', dest='assign', type=float, default=1)
parser.add_argument('--maxf', dest='maxf', type=int, default=None)
parser.add_argument('--window-length', dest='length_window', type=int, default=200)
parser.add_argument('--compute-only', dest="compute_only", action="store_true")
parser.add_argument('--base', dest="base", action="store_true")
parser.add_argument('--rescale', dest="rescale", action="store_true")

Tt = np.load("data/training/T-T1-corrected-transition_iter3.npy")

args = parser.parse_args()

filename = args.filename
length_window = args.length_window
maxf = args.maxf
weight_name = args.weight_name
typem = args.typem
extra = args.extra
root = args.root
nc = args.nc
target = args.target

ntwk = model(typem=typem, base=args.base, nc=args.nc)
ntwk.load_weights(weight_name)

if args.base:
    root = '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/'

X, y = load_data([filename], root=root, nc=args.nc)
if not args.base:
    Xr, yr, fn = load_events(X, y, min_length=length_window, base=args.base,
                             maxf=args.maxf)
    extra_e = []
else:
    Xr, yr, fn, extra_e = load_events(X, y, min_length=length_window,
                                      base=args.base, maxf=args.maxf, extra=True)
assert(len(fn) == len(Xr))
print("Nfiles", len(Xr))
yr = np.array(yr)
Xt, yt, which_keep,NotT = transform_reads(
    Xr, yr, lenv=length_window, Tt=Tt, rescale=args.rescale, extra_e=extra_e)

fn = [ifn for ifn, w in zip(fn, which_keep) if w]


Predicts = []
for xt in Xt:
    if nc == 1:
        Predicts.append(np.mean(ntwk.predict(xt)))
    if nc == 2:
        Predicts.append(np.mean(ntwk.predict(xt)[target]))

fname = os.path.split(filename)[-1][:-4]
Predicts = np.array(Predicts)
pylab.hist(Predicts, bins=40)
fnf = weight_name[:-5]+"histo_B_T_%s" % (fname)+extra+".pdf"
under = np.sum(Predicts < args.thres)/len(Predicts)
print("Percent of reads < %.1f" % args.thres,under)
val = min(args.assign / (1-under),1)

pylab.savefig(fnf)
print("Writing", weight_name[:-5]+"histo_B_T_%s" % (fname)+extra+".pdf")
csvf = pd.read_csv(filename)
wn = weight_name[:-5]
csvf[wn] = [val for _ in range(len(csvf))]
found = 0
assert len(Predicts) == len(fn)
for f, r in zip(fn, Predicts):
    if args.compute_only:
        b = r
    else:
        if r > args.thres:
            b = val
        else:
            b = 0
    fnshort = f.replace(root, "")

    which = csvf["filename"] == fnshort
    found += np.sum(which)
    csvf.loc[which, wn] = b

print("Number processed", found)
csvf.to_csv(filename, index=False)
