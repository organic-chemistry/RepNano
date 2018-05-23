import numpy as np
from .helpers import scale, scale_clean
from numba import jit
import pandas as pd


defs = {
    'r9.4': {
        'ed_params': {
            'window_lengths': [3, 6], 'thresholds': [1.4, 1.1],
            'peak_height': 0.2
        }
    },
    'r9': {
        'ed_params': {
            'window_lengths': [6, 12], 'thresholds': [2.0, 1.1],
            'peak_height': 1.2
        }
    },
    'r9.5b': {
        'ed_params': {
            'window_lengths': [4, 5], 'thresholds': [1.4, 1.0],
            'peak_height': 1
        }
    },
    'r9.5': {
        'ed_params': {
            'window_lengths': [4, 6], 'thresholds': [1.4, 1.0],
            'peak_height': 0.65
        }
    },
    'rf': {
        'ed_params': {
            'window_lengths': [4, 6], 'thresholds': [1.4, 1.1],  # [1.4, 1.1],
            'peak_height': 1.2  # 1.2
        }
    }

}


def get_raw(h5):
    # print(h5["Raw/Reads"].keys())
    rk = list(h5["Raw/Reads"].keys())[0]

    raw = h5["Raw/Reads"][rk]["Signal"]
    # print(list(h5["Raw/Reads"].keys()))
    meta = h5["UniqueGlobalKey/channel_id"].attrs
    offset = meta["offset"]
    raw_unit = meta['range'] / meta['digitisation']
    raw = (raw + offset) * raw_unit
    sl = meta["sampling_rate"]
    # print(tracking

    return raw, sl


def find_stall_old(events, threshold):
    count_above = 0
    start_ev_ind = 0
    for ev_ind, event in enumerate(events[:100]):
        if event['mean'] <= threshold:
            count_above = 0
        else:
            count_above += 1

        if count_above == 2:
            start_ev_ind = ev_ind - 1
            break

    new_start = 0
    count = 0
    for idx in range(start_ev_ind, len(events)):
        if events['mean'][idx] > threshold:
            count = 0
        else:
            count += 1

        if count == 3:
            new_start = idx - 2
            break

    return new_start


def find_stall(events, start_threshold, end_threshold, raw, sampling_rate, max_under_threshold=10):

    std = pd.Series(raw).rolling(window=20, center=False).std(
    ).rolling(window=100, center=False).mean()
    mean = pd.Series(raw).rolling(window=20, center=False).mean()
    dtav = std  # / mean

    istart = None
    iend = None
    cumul = 0
    for iel, el in enumerate(dtav):
        if not np.isnan(el) and el > start_threshold and istart is None:

            istart = iel / sampling_rate

        if istart is not None:
            if el < end_threshold:
                cumul += 1
                if cumul == 1:
                    iend = iel / sampling_rate
            else:
                cumul = 0
            if cumul == max_under_threshold:
                break

    real_start = 0
    real_end = None
    for ievent, start in enumerate(events["start"]):
        if istart is not None and start > istart and real_start is 0:
            real_start = 0 + ievent
        if iend is not None and start > iend and real_end is None:
            real_end = 0 + ievent
            break
    return real_start, real_end


def get_tstat(s, s2, wl):
    eta = 1e-100
    windows1 = np.concatenate([[s[wl - 1]], s[wl:] - s[:-wl]])
    windows2 = np.concatenate([[s2[wl - 1]], s2[wl:] - s2[:-wl]])
    means = windows1 / wl
    variances = windows2 / wl - means * means
    variances = np.maximum(variances, eta)
    delta = means[wl:] - means[:-wl]
    deltav = (variances[wl:] + variances[:-wl]) / wl
    return np.concatenate([np.zeros(wl), np.abs(delta / np.sqrt(deltav)), np.zeros(wl - 1)])


def extract_events(h5, chem, window_size=None, old=True):
    # print("ed")
    raw, sl = get_raw(h5)

    param = defs[chem]["ed_params"]
    param["old"] = old
    if window_size is not None:
        param['window_lengths'] = [window_size - 1, window_size + 1]
        print("Modif length", window_size)
    events = event_detect(raw, sl, **param)
    med, mad = med_mad(events['mean'][-100:])
    max_thresh = med + 1.48 * 2 + mad

    #first_event = find_stall(events, max_thresh)
    # first_event, last_event = find_stall(
    #    events, threshold=0.05, raw=raw, sampling_rate=sl, max_under_threshold=100)

    first_event, last_event = find_stall(
        events, start_threshold=8.5, end_threshold=4, raw=raw, sampling_rate=sl, max_under_threshold=750)
    print("First event", first_event)
    return events[:]


def med_mad(data):
    dmed = np.median(data)
    dmad = np.median(abs(data - dmed))
    return dmed, dmad


def compute_prefix_sums(data):
    data_sq = data * data
    return np.cumsum(data), np.cumsum(data_sq)


def peak_detect(short_data, long_data, short_window, long_window, short_threshold, long_threshold,
                peak_height):
    long_mask = 0
    NO_PEAK_POS = -1
    NO_PEAK_VAL = 1e100
    peaks = []
    short_peak_pos = NO_PEAK_POS
    short_peak_val = NO_PEAK_VAL
    short_found_peak = False
    long_peak_pos = NO_PEAK_POS
    long_peak_val = NO_PEAK_VAL
    long_found_peak = False

    for i in range(len(short_data)):
        val = short_data[i]
        if short_peak_pos == NO_PEAK_POS:
            if val < short_peak_val:
                short_peak_val = val
            elif val - short_peak_val > peak_height:
                short_peak_val = val
                short_peak_pos = i
        else:
            if val > short_peak_val:
                short_peak_pos = i
                short_peak_val = val
            if short_peak_val > short_threshold:
                long_mask = short_peak_pos + short_window
                long_peak_pos = NO_PEAK_POS
                long_peak_val = NO_PEAK_VAL
                long_found_peak = False
            if short_peak_val - val > peak_height and short_peak_val > short_threshold:
                short_found_peak = True
            if short_found_peak and (i - short_peak_pos) > short_window / 2:
                peaks.append(short_peak_pos)
                short_peak_pos = NO_PEAK_POS
                short_peak_val = val
                short_found_peak = False

        if i <= long_mask:
            continue
        val = long_data[i]
        if long_peak_pos == NO_PEAK_POS:
            if val < long_peak_val:
                long_peak_val = val
            elif val - long_peak_val > peak_height:
                long_peak_val = val
                long_peak_pos = i
        else:
            if val > long_peak_val:
                long_peak_pos = i
                long_peak_val = val
            if long_peak_val - val > peak_height and long_peak_val > long_threshold:
                long_found_peak = True
            if long_found_peak and (i - long_peak_pos) > long_window / 2:
                peaks.append(long_peak_pos)
                long_peak_pos = NO_PEAK_POS
                long_peak_val = val
                long_found_peak = False

    return peaks


def generate_events_old(ss1, ss2, peaks, sample_rate):
    peaks.append(len(ss1))
    events = np.empty(len(peaks), dtype=[('start', float), ('length', float),
                                         ('mean', float), ('stdv', float)])
    s = 0
    s1 = 0
    s2 = 0
    for i, e in enumerate(peaks):
        events[i]["start"] = s
        l = e - s
        events[i]["length"] = l
        m = (ss1[e - 1] - s1) / l
        events[i]["mean"] = m
        v = max(0.0, (ss2[e - 1] - s2) / l - m * m)

        events[i]["stdv"] = np.sqrt(v)
        s = e
        s1 = ss1[e - 1]
        s2 = ss2[e - 2]
    print("Generate")
    events["start"] /= sample_rate
    events["length"] /= sample_rate

    return events


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.


@jit
def find_best_partition(signal, gamma=0.1, maxlen=10, minlen=1):
    maxlen -= 1
    signal = np.array(signal)
    B = np.zeros(len(signal))
    p = np.zeros(len(signal), dtype=np.int)

    inf = 10000000000000000000
    n = len(signal)

    B[0] = -gamma

    for r in range(1, n):
        # print(r,B)
        # print(r,p)
        B[r] = inf
        for l in range(max(1, r - maxlen), r + 1 - minlen + 1):
            b = B[l - 1] + gamma + np.sum((signal[l:r + 1] - np.mean(signal[l:r + 1]))**2)
            # print(B[r])
            if b <= B[r]:
                B[r] = b
                p[r] = l - 1
    return p


@jit
def return_start_length_mean_std(partition, signal, allinfos=False):
    start = []
    length = []
    mean = []
    std = []
    r = len(signal) - 1
    l = partition[r]
    allinfo = []
    while r > 0:
        # print(l,r)
        start.append(l + 1)
        if l == 0:
            start[-1] -= 1

        length.append(r - l)
        mean.append(np.mean(signal[start[-1]:start[-1] + length[-1]]))
        std.append(np.sum((signal[start[-1]:start[-1] + length[-1]] - mean[-1])**2)**0.5)
        allinfo.append(signal[start[-1]:start[-1] + length[-1]])
        r = l
        l = partition[r]
    if allinfos:
        return start[::-1], length[::-1], mean[::-1], std[::-1], allinfo[::-1]
    else:
        return start[::-1], length[::-1], mean[::-1], std[::-1]

import time


def tv_segment(signal, gamma=0.1, maxlen=10, minlen=1, sl=6024, allinfos=False, flatten=False):
    if flatten:
        t0 = time.time()
        signal = np.array(signal) - pd.Series(signal).rolling(200,
                                                              min_periods=1, center=True).median()
        signal = np.array(signal)
        print("flatten", time.time() - t0)
    p = find_best_partition(np.array(signal, dtype=np.float32),
                            gamma=gamma, maxlen=maxlen, minlen=minlen)
    r = return_start_length_mean_std(p, signal, allinfos)
    if not allinfos:
        return pd.DataFrame({"start": np.array(r[0]) / sl, "length": np.array(r[1]) / sl, "mean": r[2], "stdv": r[3]})
    else:
        d = pd.DataFrame({"start": np.array(
            r[0]) / sl, "length": np.array(r[1]) / sl, "mean": r[2], "stdv": r[3], "all": r[4]})

        return d


def generate_events(ss1, ss2, peaks, sample_rate, raw):
    print("New")
    peaks.append(len(ss1))
    events = np.empty(len(peaks), dtype=[('start', float), ('length', float),
                                         ('mean', float), ('stdv', float)])
    s = 0

    for i, e in enumerate(peaks):
        events[i]["start"] = s
        l = e - s
        events[i]["length"] = l

        m = np.mean(raw[s:s + l])
        events[i]["mean"] = m

        events[i]["stdv"] = np.sqrt(np.mean((raw[s:s + l] - m)**2))
        #print(l, sample_rate)
        s = e
    print("Generate med")
    events["start"] /= sample_rate
    events["length"] /= sample_rate

    return events


def event_detect(raw_data, sample_rate,
                 window_lengths=[16, 40], thresholds=[8.0, 4.0], peak_height=1.0, old=True):
    """Basic, standard even detection using two t-tests

    :param raw_data: ADC values
    :param sample_rate: Sampling rate of data in Hz
    :param window_lengths: Length 2 list of window lengths across
        raw data from which `t_stats` are derived
    :param thresholds: Length 2 list of thresholds on t-statistics
    :peak_height: Absolute height a peak in signal must rise below
        previous and following minima to be considered relevant
    """
    sums, sumsqs = compute_prefix_sums(raw_data)

    tstats = []
    for i, w_len in enumerate(window_lengths):
        tstat = get_tstat(sums, sumsqs, w_len)
        tstats.append(tstat)

    peaks = peak_detect(tstats[0], tstats[1], window_lengths[0], window_lengths[1], thresholds[0],
                        thresholds[1], peak_height)
    if old:
        events = generate_events_old(sums, sumsqs, peaks, sample_rate)
    else:
        events = generate_events(sums, sumsqs, peaks, sample_rate, raw_data)

    return events
