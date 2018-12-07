from .simple_utilities import load_events, transform_reads, get_T_ou_B_delta_ind
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, AveragePooling1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import argparse
import os


def model(typem=1, window_length=None, base=False):
    init = 1
    if base:
        init = 5
    print(init)
    if typem == 1:

        if window_length is None:
            lenv = 200
        else:
            lenv = window_length
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                         activation='relu', input_shape=(lenv, init)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
        ntwk = model

    if typem == 7:
        model = Sequential()
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

    print(ntwk.summary())
    return ntwk, lenv


parser = argparse.ArgumentParser()


parser.add_argument('--weight', dest='weight_name', type=str)
parser.add_argument('--typem', dest='typem', type=int, default=7)
parser.add_argument('--maxf', dest='maxf', type=int, default=None)
parser.add_argument('--window-length', dest='length_window', type=int,
                    default=96)
parser.add_argument('--overlap', dest='overlap', type=int, default=None)
parser.add_argument('--delta', dest="delta", action="store_true")
parser.add_argument('--norescale', dest="rescale", action="store_false")
parser.add_argument('--raw', dest="raw", action="store_true")
parser.add_argument('--nobase', dest="base", action="store_false")
parser.add_argument('--verbose', dest="verbose", action="store_true")

parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--directory', type=str, default='',
                    help="Directory where read files are stored")
parser.add_argument('reads', type=str, nargs='*')


args = parser.parse_args()

length_window = args.length_window
maxf = args.maxf
weight_name = args.weight_name
typem = args.typem
reads = args.reads
directory = args.directory
output = args.output
Nmax = maxf

ntwk, lenv = model(typem=typem, window_length=length_window, base=args.base)
ntwk.load_weights(weight_name)

if length_window is None:
    length_window = lenv

Tt = np.load("data/training/T-T1-corrected-transition_iter3.npy")
Tb = np.load("data/training/B-corrected-transition_iter1.npy")


if len(reads) == 0 and len(directory) == 0:
    print("No read found")
    exit()


dire = os.path.split(output)[0]
if dire != "":
    os.makedirs(dire, exist_ok=True)
print("Writing on %s" % output)

fo = open(output, "w")
fo1 = open(output+"_ratio", "w")

files = reads
if reads == "":
    files = []
if len(directory):
    files += [os.path.join(directory, x) for x in os.listdir(directory)]
    nfiles = []
    for f in files:
        if os.path.isdir(f):
            nfiles += [os.path.join(f, x) for x in os.listdir(f)]
        else:
            nfiles.append(f)
    files = nfiles

if Nmax is not None:
    files = files[-Nmax:]

print("Found %i reads" % len(files))
for i, read in enumerate(files):
    if args.verbose:
        print(read)
    X = [read]
    y = [[0, 0]]
    if not args.base:
        Xrt, yrt, fnt = load_events(X, y, min_length=2*length_window,
                                    raw=args.raw, base=args.base,
                                    maxf=args.maxf, verbose=False)
        extra_e = []
    else:
        Xrt, yrt, fnt, extra_e = load_events(
            X, y, min_length=2*length_window, raw=args.raw,
            base=args.base, maxf=args.maxf, extra=True, verbose=False)

    print(extra_e)
    if len(Xrt) == 0:
        if args.verbose:
            print("No event or too short")
        continue
    seq2, TouB3, Success = get_T_ou_B_delta_ind(Xrt[0], Tt, Tb, True)
    if not Success:
        continue
    Xt, yt, _ = transform_reads(Xrt, np.array(yrt), lenv=length_window,
                                max_len=None, overlap=args.overlap,
                                delta=args.delta, rescale=args.rescale,
                                extra_e=extra_e, Tt=Tt)
    if len(Xt) == 0:
        continue

    if args.overlap is None:
        res = ntwk.predict(Xt[0])
    else:
        xt = np.array(Xt[0])
        # print(xt.shape)
        r = ntwk.predict(xt.reshape(-1, length_window, xt.shape[-1]))
        # print(len(r))
        r = r.reshape(args.overlap, -1, 1)

        res = np.median(r, axis=0)

    # print(res.shape)
    res0 = np.ones((res.shape[0], length_window, 1)) * res[::, np.newaxis, ::]

    res0 = res0.flatten()

    res = res0

    lc = len(res)

    seq2 = seq2[:lc]

    fo.writelines(">%s_template_deepnano %s \n" % (read, str(extra_e[0][1])))
    fo.writelines("".join(seq2) + "\n")

    fo1.writelines(">%s_template_deepnano\n" % read)
    fo1.writelines(" ".join(["%.2f" % ires2 for ires2 in res])+"\n")

fo.close()
fo1.close()
