# LSTM and CNN for sequence classification in the IMDB dataset
import _pickle as cPickle
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
from hyperopt.mongoexp import MongoTrials
import h5py
import glob
import pandas as pd
import numpy as np
from repnano.models.create_model import create_model

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from repnano.models.simple_utilities import load_data, load_events, transform_reads


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]


def load_data_complete(dataset, root, per_dataset=None, lenv=200,
                       shuffle=True, pmix=None, values=[], delta=False,
                       raw=False, rescale=False, base=False, noise=False, nc=1):

    Tt = np.load("data/training/T-T1-corrected-transition_iter3.npy")
    X_t, y_t = [], []
    for data in dataset:
        print("Loading", data)
        ws = 5
        if "T-yeast" in data:
            ws = 8

        t0 = time.time()
        X, y = load_data([data], root=root, per_dataset=per_dataset,
                         values=values+[["init_B", 0], ["init_I", 1]], nc=nc)  # X filename,y B amount

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

        Xpp, ypp, _ ,_= transform_reads(Xp, np.array(yp), lenv=lenv, delta=delta,
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
parser.add_argument('--eroot', type=str, default="./")
parser.add_argument('--cnv', dest="lstm", action="store_false")
parser.add_argument('--per-dataset', dest="per_dataset", type=int, default="400")
parser.add_argument('--pmix', dest="pmix", type=float, default=None)
parser.add_argument('--incweightT', dest="incweightT", type=float, default=None)
parser.add_argument('--delta', dest="delta", action="store_true")
parser.add_argument('--raw', dest="raw", action="store_true")
parser.add_argument('--rescale', dest="rescale", action="store_true")
parser.add_argument('--base', dest="base", action="store_true")
parser.add_argument('--bi', dest="bi", action="store_true")
parser.add_argument('--train-val', dest="train_val", action="store_true")
parser.add_argument('--noise-norm', dest="noise_norm", action="store_true")
parser.add_argument('--initw', type=str, default="")
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--on-percent', dest="on_percent", action="store_true")
parser.add_argument("--take-val-from",dest="take",default=None)
parser.add_argument('--cost', type=str, default="logcosh")
parser.add_argument('--typem', type=int, default=1)



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
      # '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T-human.csv',
        #     '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-27-human.csv',
    files = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T-human.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-27-human.csv']
    files = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-40-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/T1-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-69-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B1-yeast.csv',
             '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-9-yeast.csv']
    if args.nc == 2:
        files += ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/I-yeast.csv']
    #
    #         '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv']

    indep_val = []  # [ '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-40-yeast.csv',
    #    "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-yeast.csv"]

    val = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-9-yeast.csv']

if args.on_percent:
    root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name"
    files = glob.glob("/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/percent*.csv")
    val = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name/B-9-yeast.csv']

    indep_val = []
    #val = []
# indep_val = ["/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv"]
argparse_dict["traning"] = files
argparse_dict["indep_val"] = indep_val


with open(args.root + '/params.json', 'w') as fp:
    json.dump(argparse_dict, fp, indent=True)

init = 1
if args:
    init = 5


space = {
    'filters': hp.quniform('filters', 16, 128, 1),
    'kernel_size': hp.quniform('kernel_size', 64, 124, 1),
    'choice_pooling': hp.choice('choice_pooling', [{"pooling": False, },
                                                   {"pooling": True,
                                                    "pool_size": hp.choice("pool_size", [2, 4])}]),
    'dropout': hp.choice('dropout', [0, 0.25, 0.4]),
    'neurones': hp.quniform('neurones', 20, 300, 1),
    'batch_size': hp.quniform('batch_size', 28, 128, 1),
    'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop']),
    'activation': hp.choice('activation', ["linear", "sigmoid"])
}


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
"""
if args.initw is not None:
    model.load_weights(args.initw)

"""
rootw = os.path.join("/data/bioinfo@borvo/users/jarbona/mongo_net/first/", args.eroot)
os.makedirs(rootw, exist_ok=True)


if args.train_val:
    values=[["test_with_tombo_LSTM_alls_4000_noise_Tcorrected_iter3_filter//weights.17-0.01", 0],
        ["../mongo_net/first/B-Iweights_filters-32kernel_size-3choice_pooling-pooling-Truepool_size-2neurones-100batch_size-50optimizer-adamactivation-linearnc-2dropout-0", 1]]

    if args.take is not None:
        values.append([args.take,0])

    X_train, y_train = load_data_complete(train_test, root=root,
                                          per_dataset=args.per_dataset,
                                          lenv=lenv, pmix=args.pmix,
                                          values=values,
                                          delta=args.delta, raw=args.raw,
                                          rescale=args.rescale, base=args.base,
                                          noise=args.noise_norm, nc=args.nc)
    if val != []:
        X_val, y_val = load_data_complete(val, root=root, per_dataset=50, lenv=lenv, pmix=args.pmix,
                                          values=[
                                              ["test_with_tombo_LSTM_alls_4000_noise_Tcorrected_iter3_filter//weights.17-0.01", 0]],
                                          delta=args.delta, raw=args.raw, rescale=args.rescale,
                                          base=args.base, noise=args.noise_norm, nc=args.nc)

        # X_val = X_val[:64 * len(X_val) // 64]
        # y_val = y_val[:64 * len(y_val) // 64]

        n90 = int(len(X_train)*0.9)
        X_val = np.concatenate((X_val, X_train[n90:]), axis=0)
        y_val = np.concatenate((y_val, y_train[n90:]), axis=0)

        X_train = X_train[:n90]
        y_train = y_train[:n90]

    np.save(os.path.join(rootw, "X_train.npy"), X_train)
    np.save(os.path.join(rootw, "y_train.npy"), y_train)
    np.save(os.path.join(rootw, "X_val.npy"), X_val)
    np.save(os.path.join(rootw, "y_val.npy"), y_val)
else:
    #with open(rootw+"train.pick", "wb") as f:#
    #    cPickle.dump([X_train, y_train], f)
    # with open(rootw + "val.pick", "wb") as f:
    #    cPickle.dump([X_val, y_val], f)
    opti = False
    if opti:
        trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='lstm')
        best = fmin(create_model, space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('best: ')
        print(best)
        with open("opti-lstm.pick", "wb") as f:
            cPickle.dump(best, f)
    else:

        params = {"filters": 32, "kernel_size": 3,
                  "choice_pooling": {"pooling": True, "pool_size": 2},
                  "neurones": 100, "batch_size": 50, "optimizer": "adam",
                  "activation": "sigmoid", "nc": args.nc, "dropout": 0, "bi": args.bi,"cost":args.cost}
        create_model(params, rootw=rootw, wn=args.initw,typem=args.typem)

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
