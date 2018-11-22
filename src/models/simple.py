# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import h5py
import glob
import pandas as pd
import numpy as np


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .simple_utilities import load_data,load_events,transform_reads




def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def load_data_complete(dataset, root, per_dataset=None, lenv=200, shuffle=True):

    X_t,y_t=[],[]
    for data in dataset:
        print("Loading",data)
        ws=5
        if "T-yeast" in data:
            ws=8
        X, y = load_data([data], root=root, per_dataset=per_dataset)  # X filename,y B amount
        # X events y B amount  filtered for length < 10000
        Xp, yp,fn = load_events(X, y, min_length=None,ws=ws)
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
parser.add_argument('--cnv', dest="lstm", action="store_false")
parser.add_argument('--per-dataset', dest="per_dataset",type=int, default="400")

args = parser.parse_args()

argparse_dict = vars(args)

# sess = tf.Session(config=tf.ConfigProto(
#        intra_op_parallelism_threads=args.num_threads))

repo = Repo("./")
argparse_dict["commit"] = str(repo.head.commit)

os.makedirs(args.root, exist_ok=True)

with open(args.root + '/params.json', 'w') as fp:
    json.dump(argparse_dict, fp, indent=True)

if args.lstm:
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
    #model.load_weights("test_longueur/weights.05-0.02.hdf5")
else:
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
    model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
    #model.load_weights("test_cnv2/weights.18-0.03.hdf5")

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
         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B1-yeast.csv']
#         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv']


indep_val = ["/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv",
             "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv"]
#indep_val = files
train_test = files
per_dataset=400

for val in indep_val:
    train_test.remove(val)

"""
files = ["/home/jarbona/deepnano5bases/notebooks/exploratory/test.csv"]

indep_val = files
train_test = files

per_dataset=None
"""
print(train_test)
print(indep_val)

if args.lstm:
    lenv=200
else:
    lenv=256
X_train, y_train = load_data_complete(train_test, root=root, per_dataset=args.per_dataset, lenv=lenv)
X_val, y_val = load_data_complete(indep_val, root=root, per_dataset=50, lenv=lenv)

print(X_train.shape, y_train.shape)
X_val = X_val[:64 * len(X_val) // 64]
y_val = y_val[:64 * len(y_val) // 64]
print(y_train[::40],np.mean(y_train,axis=0))
#, validation_data=(X_val, y_val[::, 0], y_val[::, 1])
model.fit(X_train, y_train[::, 0], epochs=100, batch_size=64,
          sample_weight=y_train[::, 1], validation_split=0.1, callbacks=[checkpointer, es,reduce_lr])
# Final evaluation of the model
scores = model.evaluate(X_val, y_val[::, 0], verbose=0)
print("Accuracy: %.2f%%" % (scores))
