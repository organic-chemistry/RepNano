# LSTM and CNN for sequence classification in the IMDB dataset
import time
import tensorflow as tf
from git import Repo
import json
import os
import argparse
import numpy
from keras.models import Sequential
from keras.layers import Dense, Flatten, AveragePooling1D, TimeDistributed, Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import h5py
import glob
import pandas as pd
import numpy as np


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from .simple_utilities import load_data, load_events, transform_reads


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def load_data_complete(dataset, root, per_dataset=None, lenv=200,
                       shuffle=True, pmix=None, values=[], delta=False,
                       raw=False, rescale=False, base=False, noise=False):

    Tt = np.load("T-T1-corrected-transition_iter3.npy")
    X_t, y_t = [], []
    for data in dataset:
        print("Loading", data)
        ws = 5
        if "T-yeast" in data:
            ws = 8

        t0 = time.time()
        X, y = load_data([data], root=root, per_dataset=per_dataset,
                         values=values+["init_B"])  # X filename,y B amount

        t1 = time.time()
        print(t1-t0, "load csv")
        t0 = time.time()
        # X events y B amount  filtered for length < 10000
        if base:
            Xp, yp, fn, extra_e = load_events(
                X, y, min_length=None, ws=ws, raw=raw, base=base, extra=True)
        else:
            extra_e = []
            Xp, yp, fn = load_events(X, y, min_length=None, ws=ws, raw=raw, base=base)
        t1 = time.time()
        print(t1-t0, "load events")
        t0 = time.time()

        print("Mean Values", np.mean(yp, axis=0))
        print("Total cumulated read length", np.sum([len(xi["mean"]) for xi in Xp]))
        assert(len(Xp) == len(yp))

        Xpp, ypp, _ = transform_reads(Xp, np.array(yp), lenv=lenv, delta=delta,
                                      rescale=rescale, noise=noise, extra_e=extra_e, Tt=Tt)
        t1 = time.time()
        print(t1-t0, "transform")
        t0 = time.time()

        Xpp = np.concatenate(Xpp, axis=0)
        ypp = np.concatenate(ypp, axis=0)
        t1 = time.time()
        print(t1-t0, "concat")
        t0 = time.time()
        print("Total cumulated read length_after_cut", Xpp.shape[0])

        X_t.append(Xpp)
        y_t.append(ypp)

    # print(X_t)
    X_t = np.concatenate(X_t, axis=0)
    y_t = np.concatenate(y_t, axis=0)

    if pmix is not None:
        print("Mixing", pmix)
        a = np.arange(len(X_t))
        m1 = np.random.choice(a, int(len(a)*pmix))
        m2 = np.random.choice(a, int(len(a)*pmix))
        nX = np.concatenate((X_t[m1, :100, ::], X_t[m2, 100:, ::]), axis=1)
        ny = y_t[m1]/2+y_t[m2]/2
        X_t = np.concatenate([X_t, nX], axis=0)
        y_t = np.concatenate([y_t, ny], axis=0)

    if shuffle:
        X_t, y_t = unison_shuffled_copies(X_t, y_t)
    return X_t,  y_t

# fix random seed for reproducibility
# load the dataset but only keep the top n words, zero the rest
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="data/training/")
parser.add_argument('--cnv', dest="lstm", action="store_false")
parser.add_argument('--per-dataset', dest="per_dataset", type=int, default="400")
parser.add_argument('--pmix', dest="pmix", type=float, default=None)
parser.add_argument('--incweightT', dest="incweightT", type=float, default=None)
parser.add_argument('--delta', dest="delta", action="store_true")
parser.add_argument('--raw', dest="raw", action="store_true")
parser.add_argument('--rescale', dest="rescale", action="store_true")
parser.add_argument('--base', dest="base", action="store_true")
parser.add_argument('--noise-norm', dest="noise_norm", action="store_true")
parser.add_argument('--initw', type=str, default=None)


args = parser.parse_args()

argparse_dict = vars(args)

# sess = tf.Session(config=tf.ConfigProto(
#        intra_op_parallelism_threads=args.num_threads))

repo = Repo("./")
argparse_dict["commit"] = str(repo.head.commit)

os.makedirs(args.root, exist_ok=True)
if not args.base:
    root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw"
else:
    root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name"
files = glob.glob(root + "/*.csv")
# ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-yeast.csv',
if not args.base:
    files = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T1-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-69-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-40-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B1-yeast.csv']
    #         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv']

    indep_val = ["/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv",
                 "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv"]
else:
    files = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T-human.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-27-human.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-40-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T1-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-69-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B1-yeast.csv']
    #         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv']

    indep_val = []  # [ '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-40-yeast.csv',
    #    "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-yeast.csv"]

    val = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-9-yeast.csv']

# indep_val = ["/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv"]
argparse_dict["traning"] = files
argparse_dict["indep_val"] = indep_val


with open(args.root + '/params.json', 'w') as fp:
    json.dump(argparse_dict, fp, indent=True)

init = 1
if args:
    init = 5


space = {
    'filters': hp.uniform('filters', 16, 128),
    'kernel_size': hp.uniform('kernel_size', 64, 1024),
    'choice_pooling': hp.choice('choice_pooling', [{"pooling": False},
                                                   {"pooling": True,
                                                    "pool_size": hp.choice("pool_size": [2, 4])}]),
    'dropout': hp.choice('dropout', [0, 0.25, 0.4]),
    'neurones': hp.uniform('neurones', 20, 300),
    'batch_size': hp.uniform('batch_size', 28, 128),
    'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
    'activation': hp.choice('activation', ["linear", "sigmoid"])
}


def create_model(params):
    # typem=1, kernel_size=3, filters=32, neurones=100,
    #             activation="linear", pooling=True, mpool=2, dropout=0):
    typem = 1
    if typem == 1:
        model = Sequential()
        model.add(Conv1D(filters=params['filters'],
                         kernel_size=params['kernel_size'], padding='same',
                         activation='relu', input_shape=(160, init)))
        if params['pooling']:
            model.add(MaxPooling1D(pool_size=params["pool_size"]))
        if params['dropout'] != 0:
            model.add(Dropout(params['dropout']))

        model.add(LSTM(params['neurones']))
        model.add(Dense(1, activation=params['activation']))
        model.compile(loss='logcosh', optimizer=params['optimizer'])  # , metrics=['accuracy'])
        # model.load_weights("test_longueur_lstm_from_scratch_without_human/weights.25-0.02.hdf5")
        # model.load_weights("test_longueur/weights.05-0.02.hdf5")
    else:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu', input_shape=(160, init)))
        """
        model.add(MaxPooling1D(pool_size=4)) # 16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4)) #4
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                                 activation='relu'))

        # model.add(LSTM(100))
        # model.add(Dense(1, activation='linear'))
        """
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(AveragePooling1D(pool_size=10))
        model.add(Flatten())
        model.compile(loss='logcosh', optimizer='adam')  # , metrics=['accuracy'])
    # model.load_weights("test_cnv2/weights.18-0.03.hdf5")

        name = "".join(["%s_$s" % (p, str(value) for p, value in params.items)])
        checkpointer = ModelCheckpoint(
            filepath=args.root+'/weights_%s.hdf5' % name, verbose=1, save_best_only=True)
        es = EarlyStopping(patience=10)

        model.fit(X_train, y_train[::, 0], epochs=40, batch_size=params['batch_size'],
                  sample_weight=y_train[::, 1], validation_split=0.1, callbacks=[checkpointer, es])
        # Final evaluation of the model

        scores = model.evaluate(X_val, y_val[::, 0], verbose=0)
        print(scores)
        return {'loss': -scores, 'status': STATUS_OK}


# indep_val = files
train_test = files
per_dataset = 400

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
    lenv = 200
    lenv = 160
else:
    lenv = 256*2
    lenv = 100
    lenv = 96

if args.initw is not None:
    model.load_weights(args.initw)

X_train, y_train = load_data_complete(train_test, root=root,
                                      per_dataset=args.per_dataset,
                                      lenv=lenv, pmix=args.pmix,
                                      values=["test_with_tombo_LSTM_alls_4000_noise_Tcorrected_iter3_filter//weights.17-0.01",
                                              "test_with_tombo_CNV_logcosh_3layers/weights.22-0.01",
                                              "test_with_tombo/weights.03-0.03", "test_longueur_lstm_from_scratch_without_human_weights.25-0.02"],
                                      delta=args.delta, raw=args.raw,
                                      rescale=args.rescale, base=args.base, noise=args.noise_norm)
if val != []:
    X_val, y_val = load_data_complete(val, root=root, per_dataset=50, lenv=lenv, pmix=args.pmix,
                                      values=["test_with_tombo/weights.03-0.03",
                                              "test_longueur_lstm_from_scratch_without_human_weights.25-0.02"],
                                      delta=args.delta, raw=args.raw, rescale=args.rescale, base=args.base, noise=args.noise_norm)

    #X_val = X_val[:64 * len(X_val) // 64]
    #y_val = y_val[:64 * len(y_val) // 64]

    n90 = int(len(X_train)*0.9)
    X_val = np.concatenate((X_val, X_train[n90, :]), axis=0)
    y_val = np.concatenate((y_val, y_train[n90, :]), axis=0)


trials = Trials()
best = fmin(create_model, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best
with open("opti-lstm.pick", "w") as f:
    cPickle.dump(best, f)

"""
print(X_train.shape, y_train.shape)
print(y_train[::40], np.mean(y_train, axis=0))
print(X_train.dtype, y_train.dtype)
# for yi in y_train:
#    print(yi)
# , validation_data=(X_val, y_val[::, 0], y_val[::, 1])
if args.incweightT is not None:
    print(np.mean(y_train, axis=0))
    y_train[y_train[::, 0] == 0, 1] *= args.incweightT
    print(np.mean(y_train, axis=0))


print("Accuracy: %.2f%%" % (scores))
"""
