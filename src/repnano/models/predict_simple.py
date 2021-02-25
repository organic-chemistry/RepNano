from repnano.models.simple_utilities import load_events, transform_reads, get_T_ou_B_delta_ind, load_events_bigf
from repnano.models.create_model import create_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, AveragePooling1D
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tqdm import tqdm
import numpy as np
import argparse
import os


def model(typem=1, window_length=None, base=False, idu=False, activation="linear"):
    init = 1
    if base:
        init = 5
    print(init)
    if typem in [1, 3]:

        if window_length is None:
            lenv = 200
        else:
            lenv = window_length
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, padding='same',
                         activation='relu', input_shape=(lenv, init)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation=activation))
        model.compile(loss='mse', optimizer='adam')  # , metrics=['accuracy'])
        ntwk = model

        if idu:
            params = {"filters": 32, "kernel_size": 3,
                      "choice_pooling": {"pooling": True, "pool_size": 2},
                      "neurones": 100, "batch_size": 50, "optimizer": "adam",
                      "activation": "sigmoid", "nc": 2, "dropout": 0, "bi": False}
            ntwk = create_model(params, create_only=True, typem=args.typem)
            lenv = 160

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


def atomise(r, length_window, overlap, final_length):
    # create shift all windows by increasing factor of L
    # and do a median
    Proba = {}
    res = []
    L = int(length_window // overlap)
    rf = np.zeros((overlap, final_length)) * np.nan
    for k in range(0, overlap):
        rf[k, k * L:k * L + r.shape[1] * length_window] = np.repeat(r[k, :, 0], length_window)

    return np.nanmedian(rf, axis=0)

parser = argparse.ArgumentParser()


parser.add_argument('--weight', dest='weight_name', type=str,default=None)
parser.add_argument('--typem', dest='typem', type=int, default=7)
parser.add_argument('--maxf', dest='maxf', type=int, default=None)
parser.add_argument('--window-length', dest='length_window', type=int,
                    default=96)
parser.add_argument('--overlap', dest='overlap', type=int, default=None)
parser.add_argument('--IdU', dest='idu', action="store_true")
parser.add_argument('--activation', dest='activation', default="linear")

parser.add_argument('--delta', dest="delta", action="store_true")
parser.add_argument('--bigf', dest="bigf", action="store_true")
parser.add_argument('--norescale', dest="rescale", action="store_false")
parser.add_argument('--raw', dest="raw", action="store_true")
parser.add_argument('--nobase', dest="base", action="store_false")
parser.add_argument('--verbose', dest="verbose", action="store_true")
parser.add_argument('--percent', action="store_true")
parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--directory', type=str, default='',
                    help="Directory where read files are stored")
parser.add_argument('reads', type=str, nargs='*')




args = parser.parse_args()

abs_root = os.path.split(os.path.abspath(__file__))[0] + "/../../../"

if args.weight_name == None:
    weight_name = abs_root+"weight/test_with_tombo_CNV_logcosh_3layers_alls_4000_noise_Tcorrected_iter3_filter_weights.68-0.01.hdf5"
else:
    weight_name = args.weight_name

length_window = args.length_window
maxf = args.maxf

typem = args.typem
reads = args.reads
directory = args.directory
output = args.output
Nmax = maxf

print(args.base, args.delta, args.rescale)

ntwk, lenv = model(typem=typem, window_length=length_window, base=args.base,
                   idu=args.idu, activation=args.activation)
ntwk.load_weights(weight_name)

if length_window is None:
    length_window = lenv
if length_window != lenv:
    length_window = lenv

Tt = np.load(abs_root+"transitionM/T-T1-corrected-transition_iter3.npy")
Tb = np.load(abs_root+"transitionM/B-corrected-transition_iter1.npy")


if len(reads) == 0 and len(directory) == 0:
    print("No read found")
    exit()


dire = os.path.split(output)[0]
if dire != "":
    os.makedirs(dire, exist_ok=True)
print("Writing on %s" % output)

fo = open(output, "w")
fo1 = open(output+"_ratio_B", "w")
if args.idu:
    fo2 = open(output+"_ratio_I", "w")
if args.percent:
    prc = open(output+"_percentBrdu", "w")

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
if args.bigf:
    print("Found %i files" % len(files))
else:
    print("Found %i reads" % len(files))

Nempty_short = 0
ntreated = 0
for i, read in enumerate(files):
    if args.verbose:
        print(read)
    X = [read]
    y = [[0, 0]]

    if args.bigf:
        fun = load_events_bigf
        def fun(*args, **kwargs):
            return tqdm(load_events_bigf(*args, **kwargs))
    else:
        def fun(*args, **kwargs):
            return [load_events(*args, **kwargs)]
    for val in fun(X, y, min_length=2*length_window,
                   raw=args.raw, base=args.base,
                   maxf=args.maxf, verbose=False, extra=args.base):
        ntreated += 1
        #print(val, args.base)
        if not args.base:
            Xrt, yrt, fnt = val
            extra_e = []
        else:
            Xrt, yrt, fnt, extra_e = val

        if len(Xrt) == 0:
            Nempty_short += 1
            if args.verbose:
                print("No event or too short")
            continue
        else:
            if fnt != []:
                read = fnt[0]
        seq2, TouB3, Success = get_T_ou_B_delta_ind(Xrt[0], Tt, Tb, True)
        if not Success["success"]:
            continue
        Xt, yt, _, NotT = transform_reads(Xrt, np.array(yrt), lenv=length_window,
                                          max_len=None, overlap=args.overlap,
                                          delta=args.delta, rescale=args.rescale,
                                          extra_e=extra_e, Tt=Tt, typem=args.typem)
        # print(Xt[0])
        if len(Xt) == 0:
            continue

        if args.overlap is None:
            res = ntwk.predict(Xt[0])
            if not args.idu:
                res0 = np.ones((res.shape[0], length_window, 1)) * res[::, np.newaxis, ::]
                res0 = res0.flatten()
                Brdu = res0

            else:
                res0 = np.ones((res[0].shape[0], length_window, 1)) * res[0][::, np.newaxis, ::]
                res0 = res0.flatten()
                Brdu = res0
                res1 = np.ones((res[1].shape[0], length_window, 1)) * res[1][::, np.newaxis, ::]
                res1 = res1.flatten()
                Idu = res1
        else:
            xt = np.array(Xt[0])
            # print(xt.shape)
            r = ntwk.predict(xt.reshape(-1, length_window, xt.shape[-1]))
            # print(r)
            overlap = args.overlap
            if not args.idu:
                # print(len(r))
                Brdu = r.reshape(args.overlap, -1, 1)
                Brdu = atomise(Brdu,length_window=length_window,overlap=overlap,final_length=len(seq2))

            else:
                Brdu = r[0]
                Brdu = Brdu.reshape(args.overlap, -1, 1)
                Brdu =  atomise(Brdu,length_window=length_window,overlap=overlap,final_length=len(seq2))

                Idu = r[1]
                Idu = Idu.reshape(args.overlap, -1, 1)
                Idu = atomise(Idu,length_window=length_window,overlap=overlap,final_length=len(seq2))

        fo.writelines(">%s %s \n" % (read, str(extra_e[0][1])))
        fo.writelines("".join(seq2) + "\n")

        fo1.writelines(">%s\n" % read)
        fo1.writelines(" ".join(["%.2f" % ires2 for ires2 in Brdu])+"\n")
        if args.idu:
            fo2.writelines(">%s_template_deepnano\n" % read)
            fo2.writelines(" ".join(["%.2f" % ires2 for ires2 in Idu])+"\n")
        if args.percent:
            prc.writelines(f"{read} {np.nanmean(Brdu):.2f}\n")

        if Nmax is not None and ntreated >= Nmax:
            break
    if Nmax is not None and ntreated >= Nmax:
        break

fo.close()
fo1.close()
if args.idu:
    fo2.close()
if args.percent:
    prc.close()


print("Read empty or too short ", Nempty_short)
