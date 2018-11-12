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


def load_data(lfiles, value="init_B", root=".", per_dataset=None):
    X = []
    y = []
    for file in lfiles:
        d = pd.read_csv(file)
        X1 = [root + "/" + f for f in d["filename"]]
        y1 = d[value]
        yw = d["init_w"]

        y1 = [[iy1, iyw] for iy1, iyw in zip(y1, yw)]
        if per_dataset is None:
            X.extend(X1)
            y.extend(y1)
        else:
            X.extend(X1[:per_dataset])
            y.extend(y1[:per_dataset])
    assert (len(X) == len(y))
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


def load_events(X, y, min_length=1000):
    Xt = []
    indexes = []
    yt = []
    for ifi, filename in enumerate(X):
        # print(filename)
        h5 = h5py.File(filename, "r")
        events = get_events(h5, already_detected=False,
                            chemistry="rf", window_size=np.random.randint(5, 8), old=False, verbose=False, about_max_len=None)
        events = events[1:-1]

        if min_length is not None and len(events) < min_length:
            continue
        # print(y[ifi])
        Xt.append(events)
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
        std = events["stdv"]
        length = events["length"]

        def scale(x):
            x -= np.percentile(x, 25)
            scale = np.percentile(x, 75) - np.percentile(x, 25)
            # print(scale,np.percentile(x, 75) , np.percentile(x, 25))
            x /= scale
            if np.sum(x > 10) > len(x) * 0.05:
                print("Warning lotl of rare events")
                print(np.sum(x > 10 * scale), len(x))
            x[x > 5] = 0
            x[x < -5] = 0

            return x

        mean = scale(mean.copy())
        std = scale(std.copy())
        # print("stl")
        length = scale(length.copy())
        # print("el")
        V = np.array([mean, std, length]).T
        # print(V.shape,yi.shape)

        lc = lenv * (len(V) // lenv)
        V = V[:lc]
        V = np.array(V).reshape(-1, lenv, V.shape[-1])

        yip = np.zeros((V.shape[0], yi.shape[0]))
        yip[::] = yi
        Xt.append(V)
        yt.append(yip)
    return Xt, yt


def load_data_complete(dataset, root, per_dataset=None, lenv=200):
    X, y = load_data(dataset, root=root, per_dataset=per_dataset)  # X filename,y B amount
    # X events y B amount  filtered for length < 10000
    Xp, yp = load_events(X, y, min_length=None)
    assert(len(Xp) == len(yp))

    Xpp, ypp = transform_reads(Xp, np.array(yp), lenv=lenv)
    Xpp = np.concatenate(Xpp, axis=0)
    ypp = np.concatenate(ypp, axis=0)
    return Xp, Xpp, yp, ypp

# fix random seed for reproducibility
# load the dataset but only keep the top n words, zero the rest
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw"
files = glob.glob(root + "/*.csv")

indep_val = ["/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/B-27-human.csv",
             "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/T-human.csv"]
train_test = files
for val in indep_val:
    train_test.remove(val)


_, X_train, _, y_train = load_data_complete(train_test, root=root, per_dataset=5, lenv=200)
#_, X_test, _, y_test = load_data_complete(train_test, root=root, per_dataset=20)

print(X_train.shape, y_train.shape)
model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(200, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train[::,  0], epochs=3, batch_size=64, sample_weight=y_train[::, 1])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores))
