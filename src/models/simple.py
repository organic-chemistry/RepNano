# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import h5py
import glob
import pandas as pd
import numpy as np


from ..features.extract_events import extract_events
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def load_data(lfiles, value="init_B", root=".", per_dataset=None):
    X = []
    y = []
    for file in lfiles:
        d = pd.read_csv(file)
        #print(file, d)
        if "mean" in d.columns:
            def get_list(slist):
                slist = slist[2:-2]
                slist = slist.split(", ")
                slist = [float(si) for si in slist]
                # print(slist)
                return slist
            X1 = [get_list(f) for f in d["mean"]]
        else:
            X1 = [root + "/" + f for f in d["filename"]]

        if "savedweights.13-0.05.hdf5" in d.columns:
            print("loading from savedweights.17-0.04.hdf5")
            y1 = d["savedweights.17-0.04.hdf5"]
        else:
            y1 = d[value]
        print(np.mean(y1), np.std(y1),len(y1),len(X1))
        yw = d["init_w"]
        print("Weight", np.mean(yw),len(yw))
        y1 = [[iy1, iyw] for iy1, iyw in zip(y1, yw)]
        print(len(y1))
        if per_dataset is None:
            X.extend(X1)
            y.extend(y1)
        else:
            X.extend(X1[:per_dataset])
            y.extend(y1[:per_dataset])
    assert (len(X) == len(y))
    print(y)
    return X, y


def get_events(h5, already_detected=True, chemistry="r9.5", window_size=None,
               old=True, verbose=True, about_max_len=None):
    if already_detected:
        try:
            e = h5["Analyses/Basecall_RNN_1D_000/BaseCalled_template/Events"]
            return e
        except:
            pass
        try:
            e = h5["Analyses/Basecall_1D_000/BaseCalled_template/Events"]
            return e
        except:
            pass
    else:
        return extract_events(h5, chemistry, window_size, old=old, verbose=verbose, about_max_len=about_max_len)


def load_events(X, y, min_length=1000,ws=5):
    Xt = []
    indexes = []
    yt = []
    for ifi, filename in enumerate(X):
        # print(filename)
        if type(filename) == str:
            h5 = h5py.File(filename, "r")
            events = get_events(h5, already_detected=False,
                                chemistry="rf", window_size=np.random.randint(ws, ws+3), old=False, verbose=False, about_max_len=None)
            events = events[1:-1]

            if min_length is not None and len(events) < min_length:
                continue
            # print(y[ifi])
            Xt.append(events)
        else:
            Xt.append({"mean": filename})
        yt.append(y[ifi])
        indexes.append(ifi)
    return Xt, yt


def transform_reads(X, y, lenv=200):
    Xt = []
    yt = []
    # print(y.shape)
    # print(y)
    for events, yi in zip(X, y):
        mean = events["mean"]
        #std = events["stdv"]
        #length = events["length"]

        def scale(x):
            x -= np.percentile(x, 25)
            #scale = np.percentile(x, 75) - np.percentile(x, 25)
            # print(scale,np.percentile(x, 75) , np.percentile(x, 25))
            #x /= scale
            x /= 20
            if np.sum(x > 10) > len(x) * 0.05:
                print("Warning lotl of rare events")
                print(np.sum(x > 10 * scale), len(x))
            x[x > 5] = 0
            x[x < -5] = 0

            return x

        mean = scale(mean.copy())
        """
        mean -= np.percentile(mean, 50)
        delta = mean[1:] - mean[:-1]
        rmuninf = (delta > -15) & (delta < 15)
        mean = delta[~rmuninf]"""
        #std = scale(std.copy())
        # print("stl")
        #length = scale(length.copy())
        # print("el")
        if len(mean) < lenv:
            continue
        V = np.array([mean]).T
        # print(V.shape,yi.shape)

        lc = lenv * (len(V) // lenv)
        V = V[:lc]
        V = np.array(V).reshape(-1, lenv, V.shape[-1])

        yip = np.zeros((V.shape[0], yi.shape[0]))
        yip[::] = yi
        Xt.append(V)
        yt.append(yip)
    return Xt, yt


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def load_data_complete(dataset, root, per_dataset=None, lenv=200, shuffle=True):

    X_t,y_t=[],[]
    for data in dataset:
        ws=5
        if "T-yeast" in data:
            ws=8
        X, y = load_data(dataset, root=root, per_dataset=per_dataset)  # X filename,y B amount
        # X events y B amount  filtered for length < 10000
        Xp, yp = load_events(X, y, min_length=None,ws=ws)
        assert(len(Xp) == len(yp))

        Xpp, ypp = transform_reads(Xp, np.array(yp), lenv=lenv)
        Xpp = np.concatenate(Xpp, axis=0)
        ypp = np.concatenate(ypp, axis=0)

        X_t.append(Xpp)
        y_t.append(ypp)

    X_t = np.concatenate(X_t, axis=0)
    y_t = np.concatenate(y_t, axis=0)

    if shuffle:
        X_t, y_t = unison_shuffled_copies(X_t, y_t)
    return  X_t,  y_t

# fix random seed for reproducibility
# load the dataset but only keep the top n words, zero the rest
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


import argparse
import os
import json
from git import Repo
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="data/training/")

args = parser.parse_args()

argparse_dict = vars(args)

# sess = tf.Session(config=tf.ConfigProto(
#        intra_op_parallelism_threads=args.num_threads))

repo = Repo("./")
argparse_dict["commit"] = str(repo.head.commit)

os.makedirs(args.root, exist_ok=True)

with open(args.root + '/params.json', 'w') as fp:
    json.dump(argparse_dict, fp, indent=True)


model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                 activation='relu', input_shape=(200, 1)))
model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=32, kernel_size=5, padding='same',
#                 activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Conv1D(filters=64, kernel_size=5, padding='same',
#                 activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
model.load_weights("saved-weights.17-0.04.hdf5")
checkpointer = ModelCheckpoint(
    filepath=args.root+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
es = EarlyStopping(patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)
print(model.summary())

root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw"
files = glob.glob(root + "/*.csv")
#['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-yeast.csv',

files = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T1-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-69-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-40-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B1-yeast.csv',
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv']


indep_val = ["/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv",
             "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B1-yeast.csv"]
indep_val = files
train_test = files


for val in indep_val:
    train_test.remove(val)
files = ["/home/jarbona/deepnano5bases/notebooks/exploratory/test.csv"]
indep_val = files
train_test = files


print(train_test)
print(indep_val)
X_train, y_train = load_data_complete(train_test, root=root, per_dataset=400, lenv=200)
X_val, y_val = load_data_complete(indep_val, root=root, per_dataset=40, lenv=200)

print(X_train.shape, y_train.shape)
X_val = X_val[:64 * len(X_val) // 64]
y_val = y_val[:64 * len(y_val) // 64]
print(y_train[::40],np.mean(y_train,axis=0))
#, validation_data=(X_val, y_val[::, 0], y_val[::, 1])
model.fit(X_train, y_train[::, 0], epochs=100, batch_size=64,
          sample_weight=y_train[::, 1], validation_split=0.1, callbacks=[checkpointer, es])
# Final evaluation of the model
scores = model.evaluate(X_val, y_val[::, 0], verbose=0)
print("Accuracy: %.2f%%" % (scores))
