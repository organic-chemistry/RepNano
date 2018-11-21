import pandas as pd
from ..features.extract_events import extract_events,get_events
import h5py
import numpy as np

def load_data(lfiles, values=["saved_weights_ratio.05-0.03.hdf5","init_B"], root=".", per_dataset=None):
    X = []
    y = []
    for file in lfiles:
        d = pd.read_csv(file)
        #print(file, d)
        X1 = [root + "/" + f for f in d["filename"]]
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
    return Xt, yt,fn
