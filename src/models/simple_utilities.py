import pandas as pd
from ..features.extract_events import extract_events,get_events
import h5py
import numpy as np
import os
def load_data(lfiles, values=["saved_weights_ratio.05-0.03","init_B"], root=".", per_dataset=None):
    X = []
    y = []
    for file in lfiles:
        d = pd.read_csv(file)
        #print(file, d)
        X1 = [os.path.join(root,f) for f in d["filename"]]
        found = False
        for value in values:
            if value in d.columns:

                y1 = d[value]
                print("using value",value)
                found=True
                break
        if not found:
            print("Values not found",values)
            print("Available",d.columns)
            raise
        #print(np.mean(y1), np.std(y1),len(y1),len(X1))
        yw = d["init_w"]
        #print("Weight", np.mean(yw),len(yw))
        y1 = [[iy1, iyw] for iy1, iyw in zip(y1, yw)]
        #print(len(y1))
        if per_dataset is None:
            X.extend(X1)
            y.extend(y1)
        else:
            X.extend(X1[:per_dataset])
            y.extend(y1[:per_dataset])
    assert (len(X) == len(y))
    #print(y)
    return X, y



def load_events(X, y, min_length=1000,ws=5):
    Xt = []
    indexes = []
    yt = []
    fnames = []
    for ifi, filename in enumerate(X):
        # print(filename)

        h5 = h5py.File(filename, "r")
        events = get_events(h5, already_detected=False,
                            chemistry="rf", window_size=np.random.randint(ws, ws+3), old=False, verbose=False, about_max_len=None)
        #events = events[1:-1]

        if min_length is not None and len(events) < min_length:
            continue
        # print(y[ifi])
        Xt.append({"mean":events["mean"]})
        h5.close()

        yt.append(y[ifi])
        indexes.append(ifi)
        fnames.append(filename)
    return Xt, np.array(yt),fnames

def scale_one_read(events):
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
    V = np.array([mean]).T
    return V

def transform_reads(X, y, lenv=200,max_len=None,overlap=None):
    Xt = []
    yt = []
    # print(y.shape)
    # print(y)
    for events, yi in zip(X, y):

        V = scale_one_read(events)

        if max_len is not None:
            V = V[:max_len]

        if overlap is None:
            if lenv is not None:
                #print(V.shape,yi.shape)
                if len(V) < lenv:
                    continue
                # print(V.shape,yi.shape)

                lc = lenv * (len(V) // lenv)
                V = V[:lc]
                V = np.array(V).reshape(-1, lenv, V.shape[-1])

                yip = np.zeros((V.shape[0], yi.shape[0]))
                yip[::] = yi
            else:
                ypi=yi
        else:
            cut = int(overlap * (lenv // overlap))
            lw = cut // overlap
            lc = cut * (len(X) // cut)

            V = V[:lc]
            print(len(V))
            An = []
            for i in range(overlap - 1):
                print(lc, cut, lw, i * lw, -cut + (i + 1) * lw)
                An.append(V[i * lw:-cut + (i + 1) * lw])

            An.append(V[(overlap - 1) * lw:])

            X = np.array(An)
            print(X.shape)

        Xt.append(V)
        yt.append(yip)
    return Xt, np.array(yt)
