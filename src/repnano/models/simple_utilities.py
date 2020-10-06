import sys
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


def get_rescaled_deltas(x, TransitionM, filtered=False, rs={},thresh=0.25):
    real, th = get_signal_expected_ind(x, TransitionM)
    Tm = get_tmiddle(x)
    if rs == {}:
        # print("Comp")
        rs = rescale_deltas(real, th, Tm)

    new = real.copy()
    new = (new-rs["x"][0])/rs["x"][1]
    # print(rs["x"])
    whole, NotT, T = deltas(new, th, Tm)
    #print(NotT)
    if filtered:
        if NotT > thresh:
            return [], [], [], [],NotT
    return new, Tm, th, rs,NotT


def get_T_ou_B_delta_ind(x, TransitionT, TransitionB, filtered=False, rs={},thresh=0.25,both=False,signif=0.4,cl=None):
    new, Tm, th, rs,NotT = get_rescaled_deltas(x, TransitionT, filtered=filtered, rs=rs,thresh=thresh)
    if filtered and len(new) == 0:
        return [], [], {"success":False}

    Tm = get_strict_T_middle(x)

    # print(len(new),len(x["bases"]))
    real, thT = get_signal_expected_ind(x, TransitionT)
    real, thB = get_signal_expected_ind(x, TransitionB)

    deltasT = np.abs(new-thT)
    deltasB = np.abs(new-thB)

    significatif = (np.abs(thT-thB) > signif) # & ((deltasT<0.4) | (deltasB<0.4))
    if cl is not None:
        significatif = significatif & ((deltasT<cl) | (deltasB<cl))
    seq = x["bases"][2:-3].copy()
    which = np.argmin(np.concatenate(
        (deltasT[::, np.newaxis], deltasB[::, np.newaxis]), axis=1), axis=1)

    if both:

        #print("vanta",np.sum(significatif & Tm))
        significatif[1:] = significatif[1:] | significatif[:-1]

        which[~significatif] = 10

        which[1:] = which[1:] + which[:-1]
        #print("bf",np.sum(significatif & Tm))
        significatif[1:][which[1:]==1]=False
        which[which==1]=-1
        which[which==2]=1
        which[which==10] = 0
        which[which==11] = 1

        #print(np.sum(significatif & Tm))


    #isT = x["bases"][2:-3] == "T"
    seq[(~significatif) & Tm] = "X"
    # print(np.sum((~significatif) & Tm))

    # print(significatif)
    # print(Tm)
    seq[significatif & (which == 1) & Tm] = "B"
    TouB = [{"T": 0, "B": 1}[b] for b in seq if b in ["T", "B"]]
    return seq, TouB, {"success":True,"NotT":NotT}


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
        # delta = ((real-x[0])/x[1] - th)**2

        delta[delta > np.percentile(delta, 80)] = 0
        return np.sum(delta)
    return optimize.minimize(f, [0, 1], method="Nelder-Mead")


def deltas(which, th, Tm):
    return np.mean((which-th)**2), np.mean((which[~Tm]-th[~Tm])**2), np.mean((which[Tm]-th[Tm])**2)


def load_data(lfiles, values=[["saved_weights_ratio.05-0.03", 0],
                              ["init_B", 0],
                              ["init_I", 1]], root=".", per_dataset=None, nc=1):
    X = []
    y = []
    for file in lfiles:
        d = pd.read_csv(file)
        # print(file, d)
        X1 = [os.path.join(root, f) for f in d["filename"]]
        found = False
        for value, cat in values:
            if value in d.columns:

                y1 = d[value]
                print("using value", value)
                found = True
                break
        if not found:
            print("Values not found", values)
            print("Available", d.columns)
            raise
        # print(np.mean(y1), np.std(y1),len(y1),len(X1))
        yw = d["init_w"]
        # print("Weight", np.mean(yw),len(yw))
        yt = []
        for iy1, iyw in zip(y1, yw):
            yt.append([0, 1]*nc)
            yt[-1][2*cat] = iy1
            yt[-1][2*cat + 1] = iyw
            # print(yt[-1])
        y1 = yt
        # y1 = [[iy1, iyw] for iy1, iyw in zip(y1, yw)]
        # print(len(y1))
        if per_dataset is None:
            X.extend(X1)
            y.extend(y1)
        else:
            X.extend(X1[: per_dataset])
            y.extend(y1[: per_dataset])
    assert (len(X) == len(y))
    # print(y)
    return X, y


def load_events_bigf(X, y, min_length=1000, ws=5, raw=False, base=False,
                     maxf=None, extra=False, verbose=True):
    assert(len(X) == 1)
    h5 = h5py.File(X[0], "r")
    for v in h5.values():
        yield load_events([v], y, min_length=min_length, ws=ws, raw=raw,
                          base=base, maxf=maxf, extra=extra, verbose=verbose)


def load_events(X, y, min_length=1000, ws=5, raw=False, base=False, maxf=None, extra=False, verbose=True):
    Xt = []
    indexes = []
    yt = []
    fnames = []
    empty = 0
    extra_e = []
    for ifi, filename in enumerate(X):
        #print(filename)
        if type(filename) == str:
            h5 = h5py.File(filename, "r")
            bigf = False
        else:
            h5 = filename
            bigf = True
        tomb = False
        if base:
            tomb = True
        events, rawV, sl = get_events(h5, already_detected=False,
                                      chemistry="rf", window_size=np.random.randint(ws, ws+3),
                                      old=False, verbose=False,
                                      about_max_len=None, extra=True, tomb=tomb, bigf=bigf)
        if type(filename) != str:
            try:
                filename = filename.name
            except:
                pass

        if events is None:
            empty += 1
            events = {"mean": [], "bases": []}
        if raw:
            # print(len(rawV))
            events = {"mean": rawV}

        # events = events[1:-1]

        if min_length is not None and len(events["mean"]) < min_length:
            continue

        if extra:
            extra_e.append([rawV, sl])
        # print(y[ifi])
        if base:
            Xt.append({"mean": events["mean"], "bases": events["bases"]})
        else:
            Xt.append({"mean": events["mean"]})
        if not bigf:
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
    # x /= scale
    x /= scale
    if np.sum(x > 10) > len(x) * 0.05:
        print("Warning lotl of rare events")
        print(np.sum(x > 10), len(x))
    x[x > 5] = 0
    x[x < -5] = 0

    return x


def scale_one_read(events, rescale=False):
    mean = events["mean"]
    # std = events["stdv"]
    # length = events["length"]

    mean = scale(mean.copy(), rescale=rescale)
    """
    mean -= np.percentile(mean, 50)
    delta = mean[1:] - mean[:-1]
    rmuninf = (delta > -15) & (delta < 15)
    mean = delta[~rmuninf]"""
    # std = scale(std.copy())
    # print("stl")
    # length = scale(length.copy())
    # print("el")
    V = np.array([mean]).T
    return V

def mapb(B):
    s = "ATCG"
    r = [0, 0, 0, 0]
    r[s.index(B)] = 1
    return r


def create(X):
    r = np.zeros((len(X["mean"]),5))
    r[::,0]=X["mean"]
    for l,val in zip(["A","T","C","G"],range(1,5)):
        #print(l,val)
        r[X["bases"]==l,val]=1
    return r

def window_stack(a, stepsize=1, width=3):
    # print([[i,1+i-width or None,stepsize] for i in range(0,width)])
    return np.hstack(a[i:1+i-width or None:stepsize] for i in range(0, width))

def window_stack_numpy_v2(a,stepsize,width):
    stride = a.strides[-1]
    last_dim=a.shape[-1]
    nline = int((len(a)-width)/(stepsize) + 1)

    return np.lib.stride_tricks.as_strided(a, shape=(nline,width*last_dim), strides=(stepsize*last_dim*stride,stride))

def transform_read(X,y,window_size,pad_size):


    #V = np.array([[m] + mapb(b) for m, b in zip(X["mean"], X["bases"])])
    V=create(X)
    pad = np.zeros((pad_size,V.shape[1]))
    V = np.concatenate([pad,V,pad],axis=0)
    V = window_stack(V,stepsize=window_size-2*pad_size,width=window_size).reshape(-1, window_size,V.shape[-1])[::, ::, ::]
    return V,np.zeros(V.shape[0])+y




def transform_reads(X, y, lenv=200, max_len=None, overlap=None, delta=False,
                    rescale=False, noise=False, extra_e=[], Tt=[], typem=None):
    # print(lenv, max_len, overlap, delta,
    #          rescale, noise)
    Xt = []
    yt = []
    NotTVal = []
    # print(y.shape)
    # print(y)
    #print("Tr",typem)


    Value = {"A": 0, "T": 1, "C": 2, "G": 3, "B": 4, "I": 5}

    def embed(lv, size=6, plus_T=0):
        def se(l):
            # print(l)
            r = np.zeros(size)
            if l != "T":
                r[Value[l]] = 1
            else:
                r[Value[l]+plus_T] = 1
            return r

        # return "".join(lv)
        return np.array([se(letter) for letter in lv])

    which_keep = []
    for ip, (events, yi) in enumerate(zip(X, y)):
        #print(events)
        if typem != 3:
            if type(events) == dict and "bases" in events.keys():
                #V = np.array([[m] + mapb(b) for m, b in zip(events["mean"], events["bases"])])
                V = create(events)
                if rescale and extra_e != [] and len(V) != 0:

                    # print("Resacl")
                    new, Tm, th, rs,NotT = get_rescaled_deltas(events, Tt, filtered=True, rs={})
                    #print(NotT)
                    NotTVal.append(NotT)
                    if new != []:
                        V = V[2:2+len(new)]
                        V[::, 0] = new

                    else:
                        which_keep.append(False)
                        continue

            else:
                #print("La?")
                V = scale_one_read(events, rescale=rescale)
        if typem == 3:
            # print("la,type3")
            sc = events["bases"].copy()
            sc[sc == "B"] = "T"
            sc[sc == "I"] = "T"
            mean = events["mean"][::, np.newaxis].copy()
            mean /= 3
            V = np.concatenate((mean, embed(sc, size=4)), axis=-1)

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
    return Xt, np.array(yt), np.array(which_keep) , np.array(NotTVal)
