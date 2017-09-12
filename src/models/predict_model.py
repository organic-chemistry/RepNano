import h5py
import argparse
import os
import numpy as np
from .model import build_models
from ..features.extract_events import extract_events, scale


def get_events(h5, already_detected=True):
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
        return extract_events(h5, "r9.5")


def basecall_one_file(filename, output_file, ntwk, alph, already_detected):
    # try:
    assert(os.path.exists(filename)), "File %s does no exists" % filename
    h5 = h5py.File(filename, "r")
    events = get_events(h5, already_detected)
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
    print(np.mean(X[:, 0]))
    try:
        print("la")
        o1 = ntwk.predict(np.array(X)[np.newaxis, ::, ::])
        o1 = o1[0]

    except:
        print("ici")
        o1, o2 = ntwk.predict(X)

    # print(o1[:20])
    om = np.argmax(o1, axis=-1)
    # print(o2[:20])
    # exit()

    output = "".join(map(lambda x: alph[x], om)).replace("N", "")
    print(om.shape, len(output), len(output) / om.shape[0])

    output_file.writelines(">%s_template_deepnano\n" % filename)
    output_file.writelines(output + "\n")

    h5.close()
    return len(events)
    # except Exception as e:
    print("Read %s failed with %s" % (filename, e))
    return 0


def process(weights, Nbases, output, directory, reads=[], filter="", already_detected=True, Nmax=None):
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

        # print(Files)
        # print(files)
        # exit()
        for i, read in enumerate(files):

            if Files != []:
                if os.path.split(read)[1] not in Files:
                    continue
            print("Processing read %s" % read)
            basecall_one_file(read, fo, ntwk, alph, already_detected)

            if Nmax and i >= Nmax:
                break

        fo.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="None")
    parser.add_argument('--Nbases', type=int, choices=[4, 5], default=4)
    parser.add_argument('--output', type=str, default="output.fasta")
    parser.add_argument('--directory', type=str, default='',
                        help="Directory where read files are stored")
    parser.add_argument('reads', type=str, nargs='*')
    parser.add_argument('--detect', dest='already_detected', action='store_false')

    parser.add_argument('--filter', type=str, default='')

    args = parser.parse_args()
    # exit()
    process(weights=args.weights, Nbases=args.Nbases, output=args.output,
            directory=args.directory, reads=args.reads, filter=args.filter,
            already_detected=args.already_detected)
