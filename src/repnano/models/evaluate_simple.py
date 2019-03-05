import argparse
from repnano.models.simple_utilities import load_data, load_events, transform_reads
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, AveragePooling1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import pylab
import pandas as pd
from repnano.models.create_model import create_model


def model(typem=1, window_length=None, base=False, nc=1):
    init = 1
    if base:
        init = 5
    print(init)
    if typem in [1, 3]:

        if window_length is None:
            lenv = 200
        else:
            lenv = window_length

        """
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
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
                  "activation": "sigmoid", "nc": nc, "dropout": 0, "bi": False}
        ntwk = create_model(params, create_only=True, typem=args.typem)

    if typem == 2:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu', input_shape=(256, 1)))
        model.add(MaxPooling1D(pool_size=4))  # 64
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))  # 16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))

        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        ntwk = model
        lenv = 256
    """
        if typem == 3:
            model = Sequential()
            # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
            model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                             activation='relu', input_shape=(256, 1)))
            model.add(MaxPooling1D(pool_size=4))  # 64
            model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                             activation='relu'))
            model.add(MaxPooling1D(pool_size=4))  # 16
            model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                             activation='relu'))

            model.add(TimeDistributed(Dense(1, activation='sigmoid')))

            model.add(AveragePooling1D(pool_size=16))
            model.add(Flatten())
            ntwk = model
            lenv = 256
"""
    if typem == 4:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu', input_shape=(256*2, 1)))
        model.add(MaxPooling1D(pool_size=4))  # 64
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))  # 16
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))

        model.add(LSTM(100))
        model.add(Dense(1, activation='linear'))
        ntwk = model
        lenv = 256*2
    if typem == 5:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu', input_shape=(100, init)))
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
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(AveragePooling1D(pool_size=25))
        model.add(Flatten())
        ntwk = model
        lenv = 100

    if typem == 6:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                         activation='relu', input_shape=(100, init)))
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
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(AveragePooling1D(pool_size=25))
        model.add(Flatten())
        ntwk = model
        lenv = 100
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
        model.add(AveragePooling1D(pool_size=6))
        model.add(Flatten())
        ntwk = model
        lenv = 96
    if typem == 8:
        model = Sequential()
        # model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu', input_shape=(96, init)))
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
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same',
                         activation='relu'))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(AveragePooling1D(pool_size=6))
        model.add(Flatten())
        ntwk = model
        lenv = 96
    print(ntwk.summary())
    return ntwk, lenv


parser = argparse.ArgumentParser()

parser.add_argument('--extra', dest='extra', type=str, default="")
parser.add_argument('--root', dest='root', type=str,
                    default="/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/")
parser.add_argument('--weight-name', dest='weight_name', type=str)
parser.add_argument('--typem', dest='typem', type=int, default=1)
parser.add_argument('--nc', dest='nc', type=int, default=1)
parser.add_argument('--maxf', dest='maxf', type=int, default=None)
parser.add_argument('--window-length', dest='length_window', type=int, default=None)
parser.add_argument('--overlap', dest='overlap', type=int, default=None)
parser.add_argument('--delta', dest="delta", action="store_true")
parser.add_argument('--rescale', dest="rescale", action="store_true")
parser.add_argument('--raw', dest="raw", action="store_true")
parser.add_argument('--base', dest="base", action="store_true")


args = parser.parse_args()

length_window = args.length_window
maxf = args.maxf
weight_name = args.weight_name
typem = args.typem
extra = args.extra
root = args.root

print(args.base, args.delta, args.rescale)

ntwk, lenv = model(typem=typem, window_length=length_window, base=args.base, nc=args.nc)
ntwk.load_weights(weight_name)

if length_window is None:
    length_window = lenv

train_test = ['/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-yeast.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-human.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-human.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-yeast.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T1-yeast.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-69-yeast.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-9-yeast.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-40-yeast.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv',
              '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B1-yeast.csv']
if args.nc != 1:
    train_test.append('/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/I-yeast.csv')


if args.base:
    root = '/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/tomb/clean_name'
data = {}

Tt = np.load("data/training/T-T1-corrected-transition_iter3.npy")

for t in train_test:
    if args.base:
        t = t.replace("raw", "tomb/clean_name")
    print(t)
    ws = 5
    if "T-yeast" in t:
        # print("la")
        ws = 8
    # print(ws)
    X, y = load_data([t], root=root, values=[["test_with_tombo_CNV_logcosh_3layers/weights.22-0.01", 0],
                                             ["test_longueur_lstm_from_scratch_without_human_weights.25-0.02", 0], ["init_B", 0], ["init_I", 1]], nc=args.nc)
    if not args.base:
        Xrt, yrt, fnt = load_events(X, y, min_length=10*length_window,
                                    raw=args.raw, base=args.base, maxf=args.maxf)
        extra_e = []
    else:
        Xrt, yrt, fnt, extra_e = load_events(
            X, y, min_length=10*length_window, raw=args.raw, base=args.base,
            maxf=args.maxf, extra=True)

    # print(Xrt[0])
    if args.raw:
        max_len = 10000
    else:
        max_len = 2000
    Xt, yt, _ = transform_reads(Xrt, np.array(yrt), lenv=length_window,
                                max_len=2000, overlap=args.overlap,
                                delta=args.delta, rescale=args.rescale,
                                extra_e=extra_e, Tt=Tt, typem=args.typem)
    # print(Xt)
    # print(Xt[0])
    if args.nc == 1:
        data[t.split("/")[-1][:-4]] = [Xt, [yti[0][0] for yti in yt]]
    else:
        data[t.split("/")[-1][:-4]] = [Xt, [[yti[0][2*j] for j in range(args.nc)] for yti in yt]]


closer = ["T-yeast", "T1-yeast", "T-human", "B-9-yeast", "B-27-human",
          "B-40-yeast", "B-69-yeast", "B1-yeast", "B-yeast", "B-human"]
if args.nc != 1:
    closer.append("I-yeast")


def predict(closer, nc=1):
    Xr = []
    yr = []
    Predicts = []
    for d in closer:
        print(d)
        yr.extend(data[d][1])
        for xt in data[d][0]:
            if args.overlap is None:
                if nc == 1:
                    Predicts.append(np.mean(ntwk.predict(xt)))
                else:
                    Predicts.append([np.mean(v) for v in ntwk.predict(xt)])

            else:
                # print(xt)
                xt = np.array(xt)
                # print(xt.shape)
                r = ntwk.predict(xt.reshape(-1, length_window, xt.shape[-1]))
                # print(len(r))
                r = r.reshape(args.overlap, -1, 1)
                # print(r.shape)
                Predicts.append(np.mean(np.median(r, axis=0)))

    return Xr, np.array(yr), np.array(Predicts)


if args.nc == 1:
    Xr, yr, Predicts = predict(closer, nc=args.nc)

    closer = ["B-9-yeast", "B-yeast"]
    Xr, yr2, Predicts2 = predict(closer, nc=args.nc)

    pylab.title("Deviation %.2f, on test only (9 and B1) %.2f" %
                (np.std(Predicts-yr), np.std(Predicts2-yr2)))
    pylab.plot(Predicts)
    pylab.plot(yr, "o")
    pylab.xlabel("Sample #")
    pylab.ylabel("Ratio_b")
    filen = weight_name[:-5]+"sample_values"+extra+".pdf"
    print("Writing on %s" % filen)
    pylab.savefig(filen)
else:
    Xr, yr, Predicts = predict(closer, nc=args.nc)

    # print(Predicts)
    closer = ["B-9-yeast", "B-yeast"]
    Xr, yr2, Predicts2 = predict(closer, nc=args.nc)

    f = pylab.figure()
    nc = args.nc
    for i in range(args.nc):
        f.add_subplot(1, 2, i+1)
        # , np.std(Predicts2-yr2)))
        pylab.title("Deviation %.2f" % (np.std(Predicts[::, i]-yr[::, i])))

        pylab.plot(Predicts[::, i])
        pylab.plot(yr[::, i], "o")
        pylab.xlabel("Sample #")
        pylab.ylabel("Ratio_b")
        filen = weight_name[:-5]+"sample_values"+extra+".pdf"
    print("Writing on %s" % filen)
    pylab.savefig(filen)
