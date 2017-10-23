import h5py
import sys
import numpy as np
import os
import re
import datetime
import argparse

chars = "ACGT"
mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def scale(X, normalise_window=True):
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

    #scale = 0.0019
    ret[:, 0] = X[:, 0] * scale + shift
    ret[:, 1] = ret[:, 0]**2

    sscale = se50 / s50

    ret[:, 2] = X[:, 2] * sscale
    if normalise_window:
        print("Norm")
        #ret[:, 3] = 0.002 * ret[:, 3] / np.mean(ret[:, 3])
    print("Mean window length", np.mean(ret[:, 3]), np.std(ret[:, 3]), np.median(ret[:, 3]))
    print(scale, shift)
    print((me75 - me25) / (m75 - m25), me25 - m25, se50 / s50)

    scale_clean(X)
    return ret


def scale_clean(X, normalise_window=True):

    ret = np.array(X)

    print("std", np.mean(ret[:, 2]))
    ret[:, 0] = (X[:, 0] - 100) / 50
    ret[:, 1] = ret[:, 3] / np.mean(ret[:, 3]) - 1
    ret[:, 2] = X[:, 2] / 35

    #ret[:, 3] = 0.002 * ret[:, 3] / np.mean(ret[:, 3])
    print("Mean window scale_clean", np.mean(ret[:, : 2], axis=0), np.std(ret[:, : 2], axis=0))
    return ret[:, :3]


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
