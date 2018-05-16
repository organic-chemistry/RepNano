import h5py
import sys
import numpy as np
import os
import re
import datetime
import argparse

chars = "ACGT"
mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def scale_simple(X):

    return np.array((X["mean"][::, np.newaxis] - 100) / 15)


def scale(X, normalise_window=True):

    print("means", np.mean(X[:, : 4], axis=0))

    m25 = np.percentile(X[:, 0], 25)
    m75 = np.percentile(X[:, 0], 75)
    s50 = np.median(X[:, 2])
    me25 = 0.07499809
    me75 = 0.26622871
    se50 = 0.6103758
    ret = np.array(X)
    scale = (me75 - me25) / (m75 - m25)
    m25 *= scale
    shift = me25 - m25

    # scale = 0.0019
    ret[:, 0] = X[:, 0] * scale + shift
    ret[:, 1] = ret[:, 0]**2

    sscale = se50 / s50

    ret[:, 2] = X[:, 2] * sscale
    if normalise_window:
        print("Norm")
        # ret[:, 3] = 0.002 * ret[:, 3] / np.mean(ret[:, 3])
    #print("Mean window length")#
    print("means", np.mean(ret[:, : 4], axis=0))
    print("std", np.std(ret[:, : 4], axis=0))  # , np.std(ret[:, 3]), np.median(ret[:, 3]))
    # print(scale, shift)
    # print((me75 - me25) / (m75 - m25), me25 - m25, se50 / s50)

    # scale_clean(X)
    return ret


def scale_named(X, normalise_window=True):
    return scale(np.array([X["mean"], X["mean"]**2, X["stdv"], X["length"]]).T)


def scale_named2(X, normalise_window=True):
    return scale_clean2(np.array([X["mean"], X["mean"]**2, X["stdv"], X["length"]]).T)


def scale_named3(X, normalise_window=True):
    return scale_clean3(np.array([X["mean"], X["mean"]**2, X["stdv"], X["length"]]).T)


def scale_named4(X, normalise_window=True, maxleninf=35, maxi=5):
    Xd = np.zeros((len(X), maxleninf))
    iis = 0
    # print(X.columns)
    tot = []
    for s in X["all"]:
        tot.extend(s)

    #tot = np.array(tot)
    med = np.median(tot)
    std = np.percentile(np.array(tot) - med, 75) - np.percentile(np.array(tot) - med, 25)

    ncut = 0

    for s in X["all"]:
        Xd[iis][:len(s)] = s
        iis += 1
    z = Xd == 0
    Xd = (Xd - med) / std
    Xd[z] = 0
    ncut = np.sum(Xd > maxi) + np.sum(Xd < -maxi)
    Xd[Xd > maxi] = maxi
    Xd[Xd < -maxi] = -maxi
    # - med) / std
    #ncut += sum(Xd[iis] > maxi)
    #ncut += sum(Xd[iis] < -maxi)
    #Xd[iis][Xd[iis] > maxi] = maxi
    #Xd[iis][Xd[iis] < -maxi] = -maxi
    #

    print("Median", med, std, ncut, len(tot))
    return Xd


def scale_named4s(X, normalise_window=True, maxleninf=35):
    Xd = np.zeros((len(X), maxleninf))
    iis = 0
    # print(X.columns)
    for s in X["all"]:
        Xd[iis][:len(s)] = s
        iis += 1

    print("med", np.median(Xd))

    Xd -= np.median(Xd)
    return Xd


def scale_clean(X, normalise_window=True):

    ret = np.array(X)

    # print("std", np.mean(ret[:, 2]))
    ret[:, 0] = (X[:, 0] - 100) / 50
    ret[:, 1] = ret[:, 3] / np.mean(ret[:, 3]) - 1
    ret[:, 2] = X[:, 2] / 35

    # ret[:, 3] = 0.002 * ret[:, 3] / np.mean(ret[:, 3])
    print("Mean window scale_clean", np.mean(ret[:, : 4], axis=0), np.std(ret[:, : 4], axis=0))

    # scale_clean_two(X)
    return ret[:, : 3]


def scale_clean2(X, normalise_window=True, window=500):

    ret = np.array(X)

    # print("std", np.mean(ret[:, 2]))
    # p75 = pd.Series(ret[0]).rolling(window, center=True, min_periods=1).quantile(0.75)
    p75 = pd.Series(ret[::, 0]).quantile(0.75)

    ret[:, 0] = (ret[::, 0] - p75) / p75

    # p75 = pd.Series(ret[2]).rolling(window, center=True, min_periods=1).quantile(0.75)
    p75 = pd.Series(ret[2]).quantile(0.75)

    ret[:, 1] = (ret[::, 2] - p75) / p75

    return ret[:, : 2]


def scale_clean3(X, normalise_window=True, window=500):

    ret = np.array(X)

    # print("std", np.mean(ret[:, 2]))
    # p75 = pd.Series(ret[0]).rolling(window, center=True, min_periods=1).quantile(0.75)
    p75 = pd.Series(ret[::, 0]).quantile(0.75)

    ret[:, 0] = (ret[::, 0] - p75) / p75

    # p75 = pd.Series(ret[2]).rolling(window, center=True, min_periods=1).quantile(0.75)
    p75 = pd.Series(ret[2]).quantile(0.75)

    ret[:, 1] = (ret[::, 2] - p75) / p75

    p75 = pd.Series(ret[3]).mean()

    ret[:, 2] = (ret[::, 3] - p75) / p75

    return ret[:, : 3]


"""
def scale_clean_two(X, normalise_window=True):

    ret = np.array(X)

    print("std", np.mean(ret[:, 2]))

    ret[:, 0] = X[:, 0] - np.mean(X[:, 0])
    ret[:, 0] = ret[:, 0] / np.std(ret[:, 0])

    ret[:, 1] = X[:, 2] - np.mean(X[:, 2])
    ret[:, 1] = ret[:, 1] / np.std(ret[:, 1])

    ret[:, 2] = X[:, 3] - np.mean(X[:, 3]) + 0.001 * np.random.rand()
    ret[:, 2] = ret[:, 2] / np.std(ret[:, 2])

    ret = ret[:, :3]
    print(ret.shape)
    # ret[:, 3] = 0.002 * ret[:, 3] / np.mean(ret[:, 3])
    print("Mean window scale_clean_two", np.mean(ret[:, : 3], axis=0), np.std(ret[:, : 3], axis=0))
    return ret[:, :3]
"""
import pandas as pd


def scale_clean_two_pd(X):
    return scale_clean_two(np.array([X["mean"], X["mean"]**2, X["stdv"], X["length"]]).T)


def scale_clean_two(X, normalise_window=True, nw=100):

    ret = np.array(X)

    print("std", np.std(ret[:, 0]), np.std(ret[:, 2]), np.std(ret[:, 3]))
    print("mean", np.mean(ret[:, 0]), np.mean(ret[:, 2]), np.mean(ret[:, 3]))

    # print(pd.rolling_mean(X[:, 0], nw))

    if nw is None:
        minus = np.mean(X[:, 0])
    else:
        minus = pd.Series(X[:, 0]).rolling(nw, center=True, min_periods=1).median()

    ret[:, 0] = X[:, 0] - minus
    # print(np.std(ret[:, 0]))
    ret[:, 0] = ret[:, 0] / np.std(ret[:, 0])
    # ret[:, 0] = ret[:, 0] / np.std(ret[:, 0])

    if nw is None:
        minus = np.mean(X[:, 2])
    else:
        minus = pd.Series(X[:, 2]).rolling(nw, center=True, min_periods=1).median()

    ret[:, 1] = X[:, 2] - minus
    ret[:, 1] = ret[:, 1] / np.std(ret[:, 1])

    if nw is None:
        minus = np.mean(X[:, 3])
    else:
        minus = pd.Series(X[:, 3]).rolling(nw, center=True, min_periods=1).median()

    ret[:, 2] = X[:, 3] - minus  # + 0.001 * np.random.rand()
    ret[:, 2] = ret[:, 2] / np.std(ret[:, 2])

    ret = ret[:, : 3]
    print(ret.shape)
    # ret[:, 3] = 0.002 * ret[:, 3] / np.mean(ret[:, 3])
    print("Mean window scale_clean_two", np.mean(ret[:, : 3], axis=0), np.std(ret[:, : 3], axis=0))
    return ret[:, : 3]


def preproc_event(mean, std, length):
    mean = mean / 100.0 - 0.66
    std = std - 1
    return [mean, mean * mean, std, length]


def get_base_loc(h5):
    base_loc = "Analyses/Basecall_2D_000"
    try:
        events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
    except:
        base_loc = "Analyses/Basecall_1D_000"
    return base_loc


def extract_scaling(h5, read_type, base_loc):
    scale = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["scale"]
    scale_sd = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["scale_sd"]
    shift = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["shift"]
    drift = h5[base_loc + "/Summary/basecall_1d_" + read_type].attrs["drift"]
    return scale, scale_sd, shift, drift
