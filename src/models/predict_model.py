import h5py
import argparse
import os
import numpy as np
from .model import build_models


def scale(X):
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
    ret[:, 0] = X[:, 0] * scale + shift
    ret[:, 1] = ret[:, 0]**2

    sscale = se50 / s50

    ret[:, 2] = X[:, 2] * sscale
    return ret


def get_events(h5):
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


def basecall_one_file(filename, output_file, ntwk, alph):
    # try:
    assert(os.path.exists(filename)), "File %s does no exists" % filename
    h5 = h5py.File(filename, "r")
    events = get_events(h5)
    if events is None:
        print("No events in file %s" % filename)
        h5.close()
        return 0

    if len(events) < 300:
        print("Read %s too short, not basecalling" % filename)
        h5.close()
        return 0

    # print(len(events))
    events = events[1:-1]
    mean = events["mean"]
    std = events["stdv"]
    length = events["length"]
    X = scale(np.array(np.vstack([mean, mean * mean, std, length]).T, dtype=np.float32))
    try:
        o1 = ntwk.predict(np.array(X)[np.newaxis, ::, ::])
        o1 = o1[0]

    except:
        o1, o2 = ntwk.predict(X)

    # print(o1[:20])
    om = np.argmax(o1, axis=-1)
    # print(o2[:20])
    # exit()

    output = "".join(map(lambda x: alph[x], om)).replace("N", "")
    print(om.shape, len(output))

    output_file.writelines(">%s_template_deepnano\n" % filename)
    output_file.writelines(output + "\n")

    h5.close()
    return len(events)
    # except Exception as e:
    print("Read %s failed with %s" % (filename, e))
    return 0


def process(weights, Nbases, output, directory, reads=[], filter=""):
    assert len(reads) != 0 or len(directory) != 0, "Nothing to basecall"

    alph = "ACGTN"
    if Nbases == 5:
        alph = "ACGTBN"

    import sys
    sys.path.append("../training/")

    ntwk, _ = build_models(20)
    assert(os.path.exists(weights)), "Weights %s does not exist" % weights
    ntwk.load_weights(weights)
    print("loaded")

    Files = []
    if filter != "":
        assert(os.path.exists(filter)), "Filter %s does not exist" % filter
        with open(filter, "r") as f:
            for line in f.readlines():
                Files.append(line.split()[0])

    if len(reads) or len(directory) != 0:
        dire = os.path.split(output)[0]
        os.makedirs(dire, exist_ok=True)
        fo = open(output, "w")

        files = reads
        if reads == "":
            files = []
        if len(directory):
            files += [os.path.join(directory, x) for x in os.listdir(directory)]

        # print(Files)
        # print(files)
        # exit()
        for i, read in enumerate(files):

            if Files != []:
                if os.path.split(read)[1] not in Files:
                    continue
            print("Processing read %s" % read)
            basecall_one_file(read, fo, ntwk, alph)

        fo.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="None")
    parser.add_argument('--Nbases', choices=["4", "5"], default='4')
    parser.add_argument('--output', type=str, default="output.fasta")
    parser.add_argument('--directory', type=str, default='',
                        help="Directory where read files are stored")
    parser.add_argument('reads', type=str, nargs='*')

    parser.add_argument('--filter', type=str, default='')

    args = parser.parse_args()

    process(weights=args.weights, Nbases=args.Nbases, output=args.output,
            directory=args.directory, reads=args.reads, filter=args.filter)
