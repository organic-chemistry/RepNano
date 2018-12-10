from scipy import optimize
import pandas as pd
from ..features.extract_events import extract_events, get_events
import h5py
import numpy as np
import os

import itertools


def give_ratio_index(seq):
    val = np.zeros(len(seq))*np.nan
    val[np.array(seq) == "T"] = 0
    val[np.array(seq) == "B"] = 1
    index = [ind for ind, v in enumerate(val) if v in [0, 1]]
    return index, val[~np.isnan(val)]


def get_rescaled_deltas(x, TransitionM, filtered=False, rs={}):
    real, th = get_signal_expected_ind(x, TransitionM)
    Tm = get_tmiddle(x)
    if rs == {}:
        # print("Comp")
        rs = rescale_deltas(real, th, Tm)

    new = real.copy()
    new = (new-rs["x"][0])/rs["x"][1]
    # print(rs["x"])
    if filtered:
        whole, NotT, T = deltas(new, th, Tm)
        if NotT > 0.25:
            return [], [], [], []
    return new, Tm, th, rs


def get_T_ou_B_delta_ind(x, TransitionT, TransitionB, filtered=False, rs={}):
    new, Tm, th, rs = get_rescaled_deltas(x, TransitionT, filtered=filtered, rs=rs)
    if filtered and len(new) == 0:
        return [], [], False

    Tm = get_strict_T_middle(x)
    # print(len(new),len(x["bases"]))
    real, thT = get_signal_expected_ind(x, TransitionT)
    real, thB = get_signal_expected_ind(x, TransitionB)

    deltasT = np.abs(new-thT)
    deltasB = np.abs(new-thB)

    significatif = (np.abs(thT-thB) > 0.4)  # & ((deltasT<0.4) | (deltasB<0.4))

    seq = x["bases"][2:-3].copy()
    seq[(~significatif) & Tm] = "X"
    #print(np.sum((~significatif) & Tm))
    which = np.argmin(np.concatenate(
        (deltasT[::, np.newaxis], deltasB[::, np.newaxis]), axis=1), axis=1)
    # print(significatif)
    # print(Tm)
    seq[significatif & (which == 1) & Tm] = "B"
    TouB = [{"T": 0, "B": 1}[b] for b in seq if b in ["T", "B"]]
    return seq, TouB, True


def list_transition(length=5):
    lt = [list(s) for s in itertools.product(["A", "T", "C", "G"], repeat=length)]
    return lt, {"".join(s): lt.index(s) for s in lt}


list_trans, d_trans = list_transition(5)


def get_signal_expected(x, Tt):
    real = []
    th = np.zeros((len(x["bases"])-5))
    real = np.zeros((len(x["bases"])-5))

    for n in range(2, len(x["bases"])-3):
        delta = x["mean"][n+1]-x["mean"][n]

        i1 = d_trans["".join(x["bases"][n-2:n+3].tolist())]
        i2 = d_trans["".join(x["bases"][n-1:n+4].tolist())]
        real[n-2] = delta
        th[n-2] = Tt[i1, i2]
    return real, th


def get_indexes(x):
    number = np.zeros(len(x["bases"]), dtype=np.int)
    number[x["bases"] == "T"] = 1
    number[x["bases"] == "C"] = 2
    number[x["bases"] == "G"] = 3
    indexes = np.zeros((len(number)-4), dtype=np.int)
    for i in range(5):
        # print(len(indexes),len(number[i:len(number)-5+i]))
        indexes += number[i:len(number)-4+i]*4**i

    return indexes


def get_signal_expected_ind(x, Tt):
    real = []

    th = np.zeros((len(x["bases"])-5))
    real = np.zeros((len(x["bases"])-5))
    indexes = get_indexes(x)
    # print(len(th),len(th2))
    Plat = x["mean"][2:-2]

    # print(len(th),len(th2))
    return Plat[:-1]-Plat[1:], Tt[indexes[:-1], indexes[1:]]


def get_tmiddle(x, d=3):
    Tm = []
    for n in range(2, len(x["bases"])-d):
        if x["bases"][n] == "T" or x["bases"][n+1] == "T":
            Tm.append(True)
        else:
            Tm.append(False)
    return np.array(Tm)


def get_strict_T_middle(x, d=3):
    Tm = []
    for n in range(2, len(x["bases"])-d):
        if x["bases"][n] == "T":
            Tm.append(True)
        else:
            Tm.append(False)
    return np.array(Tm)


def rescale_deltas(real, th, Tm):

    def f(x):
        delta = ((real[~Tm]-x[0])/x[1] - th[~Tm])**2
        #delta = ((real-x[0])/x[1] - th)**2

        delta[delta > np.percentile(delta, 80)] = 0
        return np.sum(delta)
    return optimize.minimize(f, [0, 1], method="Nelder-Mead")


def deltas(which, th, Tm):
    return np.mean((which-th)**2), np.mean((which[~Tm]-th[~Tm])**2), np.mean((which[Tm]-th[Tm])**2)


def load_data(lfiles, values=["saved_weights_ratio.05-0.03", "init_B"], root=".", per_dataset=None):
    X = []
    y = []
    for file in lfiles:
        d = pd.read_csv(file)
        #print(file, d)
        X1 = [os.path.join(root, f) for f in d["filename"]]
        found = False
        for value in values:
            if value in d.columns:

                y1 = d[value]
                print("using value", value)
                found = True
                break
        if not found:
            print("Values not found", values)
            print("Available", d.columns)
            raise
        #print(np.mean(y1), np.std(y1),len(y1),len(X1))
        yw = d["init_w"]
        #print("Weight", np.mean(yw),len(yw))
        y1 = [[iy1, iyw] for iy1, iyw in zip(y1, yw)]
        # print(len(y1))
        if per_dataset is None:
            X.extend(X1)
            y.extend(y1)
        else:
            X.extend(X1[:per_dataset])
            y.extend(y1[:per_dataset])
    assert (len(X) == len(y))
    # print(y)
    return X, y


def load_events(X, y, min_length=1000, ws=5, raw=False, base=False, maxf=None, extra=False, verbose=True):
    Xt = []
    indexes = []
    yt = []
    fnames = []
    empty = 0
    extra_e = []
    for ifi, filename in enumerate(X):
        # print(filename)

        h5 = h5py.File(filename, "r")
        tomb = False
        if base:
            tomb = True
        events, rawV, sl = get_events(h5, already_detected=False,
                                      chemistry="rf", window_size=np.random.randint(ws, ws+3),
                                      old=False, verbose=False,
                                      about_max_len=None, extra=True, tomb=tomb)

        if events is None:
            empty += 1
            events = {"mean": [], "bases": []}
        if raw:
            # print(len(rawV))
            events = {"mean": rawV}

        #events = events[1:-1]

        if min_length is not None and len(events["mean"]) < min_length:
            continue

        if extra:
            extra_e.append([rawV, sl])
        # print(y[ifi])
        if base:
            Xt.append({"mean": events["mean"], "bases": events["bases"]})
        else:
            Xt.append({"mean": events["mean"]})

        h5.close()

        yt.append(y[ifi])
        indexes.append(ifi)
        fnames.append(filename)
        if maxf is not None:
            if ifi - empty > maxf:
                break
    if verbose:
        print("N empty %i, N files %i" % (empty, ifi))
    if not extra:
        return Xt, np.array(yt), fnames
    else:
        return Xt, np.array(yt), fnames, extra_e


def scale(x, rescale=False):
    x -= np.percentile(x, 25)
    if rescale:

        scale = np.percentile(x, 60) - np.percentile(x, 20)
        # print("la")
    else:
        scale = 20
    # print(scale,np.percentile(x, 75) , np.percentile(x, 25))
    #x /= scale
    x /= scale
    if np.sum(x > 10) > len(x) * 0.05:
        print("Warning lotl of rare events")
        print(np.sum(x > 10), len(x))
    x[x > 5] = 0
    x[x < -5] = 0

    return x


def scale_one_read(events, rescale=False):
    mean = events["mean"]
    #std = events["stdv"]
    #length = events["length"]

    mean = scale(mean.copy(), rescale=rescale)
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


def transform_reads(X, y, lenv=200, max_len=None, overlap=None, delta=False, rescale=False, noise=False, extra_e=[], Tt=[]):
    Xt = []
    yt = []
    # print(y.shape)
    # print(y)

    def mapb(B):
        s = "ATCG"
        r = [0, 0, 0, 0]
        r[s.index(B)] = 1
        return r
    which_keep = []
    for ip, (events, yi) in enumerate(zip(X, y)):
        if type(events) == dict and "bases" in events.keys():
            V = np.array([[m] + mapb(b) for m, b in zip(events["mean"], events["bases"])])
            if rescale and extra_e != [] and len(V) != 0:

                # print("Resacl")
                new, Tm, th, rs = get_rescaled_deltas(events, Tt, filtered=True, rs={})

                if new != []:
                    V = V[2:2+len(new)]
                    V[::, 0] = new
                else:
                    which_keep.append(False)
                    continue

        else:
            V = scale_one_read(events, rescale=rescale)

        if delta:
            V = V[1:]-V[:-1]

        if max_len is not None:
            V = V[:max_len]

        if overlap is None:
            if lenv is not None:
                # print(V.shape,yi.shape)
                if len(V) < lenv:
                    which_keep.append(False)
                    continue
                # print(V.shape,yi.shape)

                lc = lenv * (len(V) // lenv)
                V = V[:lc]
                V = np.array(V).reshape(-1, lenv, V.shape[-1])

                yip = np.zeros((V.shape[0], yi.shape[0]))
                yip[::] = yi
            else:
                ypi = yi
        else:
            lw = int(lenv // overlap)

            An = []
            for i in range(overlap - 1):
                An.append(V[i * lw:])

            An.append(V[(overlap - 1) * lw:])

            minl = np.min([len(r)//lenv for r in An])
            An = np.array([np.array(ian[:int(minl*lenv)]).reshape(-1, lenv, V.shape[-1])
                           for ian in An])
            V = np.array(An)

            yip = np.zeros((V.shape[0], yi.shape[0]))
            yip[::] = yi

        if noise:
            # print(V.shape)
            if overlap:
                V[::, ::, ::, 0] *= (0.9+0.2*np.random.rand(V.shape[1])[np.newaxis, ::, np.newaxis])
            else:
                V[::, ::, 0] *= (0.9+0.2*np.random.rand(V.shape[0])[::, np.newaxis])

        Xt.append(V)
        yt.append(yip)
        which_keep.append(True)
    which_keep = np.array(which_keep)
    assert np.sum(which_keep) == len(Xt)
    assert len(which_keep) == len(X)
    return Xt, np.array(yt), np.array(which_keep)
