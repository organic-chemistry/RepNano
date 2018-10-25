import h5py
import argparse
import os
import numpy as np
from .model import build_models
from ..features.extract_events import extract_events, scale, scale_clean
from ..features.helpers import scale_clean_two


def get_events(h5, already_detected=True, chemistry="r9.5", window_size=None, old=True):
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
        return extract_events(h5, chemistry, window_size, old=old)


def basecall_one_file(filename, output_file, ntwk, alph, already_detected,
                      n_input=1, filter_size=None, chemistry="r9.5", window_size=None, clean=False, old=True, cut=None, thres=0.5):
    # try:
    assert(os.path.exists(filename)), "File %s does no exists" % filename
    h5 = h5py.File(filename, "r")
    events = get_events(h5, already_detected, chemistry, window_size, old=old)
    if events is None:
        print("No events in file %s" % filename)
        h5.close()
        return 0

    if len(events) < 30:
        print("Read %s too short, not basecalling" % filename)
        h5.close()
        return 0

    # print(len(events))
    events = events[1:-1]
    mean = events["mean"]
    std = events["stdv"]
    length = events["length"]
    if clean:

        X = scale_clean_two(
            np.array(np.vstack([mean, mean * mean, std, length]).T, dtype=np.float32))
    else:
        X = scale(np.array(np.vstack([mean, mean * mean, std, length]).T, dtype=np.float32))
    # return
    if cut is None:
        print(X.shape)
        if n_input == 2:
            X = []
            for m, s, l in zip(mean, std, length):
                X.append([m, m * m, s, l])
                X.append([m, m * m, s, l])
            X = scale(np.array(X))

        # print(np.mean(X[:, 0]))

        p = ntwk.predict(X[np.newaxis, ::, ::])
        # print(p[0, 50:60])
        # print(X[50:60])
        # print(np.max(p[0, ::, :5]))
        if len(p) == 2:
            o1, o2 = p
            o1m = (np.argmax(o1[0], -1))
            o2m = (np.argmax(o2[0], -1))
            om = np.vstack((o1m, o2m)).reshape((-1,), order='F')
            print("len", len(om))
        else:
            o1 = p[0]
            om = np.argmax(o1, axis=-1)
            """
            rBT = o1[::, 4] / o1[::, 3]
            d = 20
            r = (1 / d < rBT) & (rBT < d) & ((om == 3) | (om == 4))

            om[r] = 6"""
    else:
        if len(X) > cut:
            lc = cut * (len(X) // cut)
            X = X[:lc]
            # print(X.shape)
            X = np.array(X).reshape(-1, cut, X.shape[-1])
        else:
            X = np.array(X)[np.newaxis, ::, ::]
        # print(X.shape)
        o1 = ntwk.predict(X)

        o1 = o1.reshape(-1, o1.shape[-1])

        ptb = o1[::, -2] / (o1[::, -3] + o1[::, -2])

        om1 = np.argmax(o1, axis=-1)

        toub = (om1 == 3) | (om1 == 4)
        #ptb = ptb[:len(toub)]
        om1[toub & (ptb > thres)] = 4
        om1[toub & (ptb < thres)] = 3

        om = om1

    # print(o1[:20])
    # print(o2[:20])
    # exit()

    output = "".join(map(lambda x: str(alph + "T")[x], om)).replace("N", "")

    print(om.shape, len(output), len(output) /
          om.shape[0], output.count("B") / (1 + output.count("T")
                                            ), output.count("L") / (1 + output.count("T")),
          output.count("E") / (1 + output.count("T")), output.count("I") / (1 + output.count("T")))
    print(output.count("T"), output.count("B"), output.count(
        "L"), output.count("E"), output.count("I"), output.count("U"))

    if filter_size is not None and len(output) < filter_size:
        print("Out too small", filter_size)
        return 0
    output_file.writelines(">%s_template_deepnano\n" % filename)
    output_file.writelines(output + "\n")

    h5.close()
    return len(events)
    # except Exception as e:
    print("Read %s failed with %s" % (filename, e))
    return 0


def process(weights, Nbases, output, directory, reads=[], filter="",
            already_detected=True, Nmax=None, size=20, n_output_network=1, n_input=1, filter_size=None,
            chemistry="r9.5", window_size=None, clean=False, old=True, res=False, attention=False, cut=None, thres=0.5):
    assert len(reads) != 0 or len(directory) != 0, "Nothing to basecall"

    alph = "ACGTN"
    if Nbases == 5:
        alph = "ACGTBN"
    if Nbases == 8:
        alph = "ACGTBLEIN"

    import sys
    sys.path.append("../training/")

    n_feat = 4
    if clean:
        n_feat = 3
    ntwk, _ = build_models(size, Nbases - 4, n_output=n_output_network, n_feat=n_feat,
                           res=res, attention=attention)
    assert(os.path.exists(weights)), "Weights %s does not exist" % weights
    ntwk.load_weights(weights)
    print("loaded")

    Files = []
    if filter != "" and filter is not None:
        assert(os.path.exists(filter)), "Filter %s does not exist" % filter
        with open(filter, "r") as f:
            for line in f.readlines():
                Files.append(line.split()[0])

    if len(reads) or len(directory) != 0:
        dire = os.path.split(output)[0]
        if dire != "":
            os.makedirs(dire, exist_ok=True)
        print("Writing on %s" % output)

        fo = open(output, "w")

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

        # print(Files)
        # print(files)
        # exit()
        if Nmax != None:
            files = files[-Nmax:]
        for i, read in enumerate(files):

            if Files != []:
                if os.path.split(read)[1] not in Files:
                    continue
            print("Processing read %s" % read)
            basecall_one_file(read, fo, ntwk, alph, already_detected,
                              n_input=n_input, filter_size=filter_size,
                              chemistry=chemistry, window_size=window_size, clean=clean, old=old, cut=cut, thres=thres)

        fo.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="None")
    parser.add_argument('--Nbases', type=int, choices=[4, 5, 8], default=4)
    parser.add_argument('--output', type=str, default="output.fasta")
    parser.add_argument('--directory', type=str, default='',
                        help="Directory where read files are stored")
    parser.add_argument('reads', type=str, nargs='*')
    parser.add_argument('--detect', dest='already_detected', action='store_false')

    parser.add_argument('--filter', type=str, default='')
    parser.add_argument('--filter-size', dest="filter_size", type=int, default=None)
    parser.add_argument('--chemistry', type=str, default='r9.5')
    parser.add_argument('--window-size', type=int, default=6, dest="window_size")
    parser.add_argument('--not-old',  dest="old", action="store_false")
    parser.add_argument('--res', dest="res", action="store_true")
    parser.add_argument('--clean', dest="clean", action="store_true")

    parser.add_argument('--attention', dest="attention", action="store_true")
    parser.add_argument('--size', dest="size", type=int, default=20)
    parser.add_argument('--cut', dest="cut", type=int, default=None)
    parser.add_argument('--thres', dest="thres", type=float, default=0.5)

    args = parser.parse_args()
    # exit()
    process(weights=args.weights, Nbases=args.Nbases, output=args.output,
            directory=args.directory, reads=args.reads, filter=args.filter,
            already_detected=args.already_detected, filter_size=args.filter_size, size=args.size,
            chemistry=args.chemistry, window_size=args.window_size,
            old=args.old, res=args.res, attention=args.attention, clean=args.clean, cut=args.cut, thres=args.thres)
