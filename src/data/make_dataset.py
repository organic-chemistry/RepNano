import os
import h5py
from ..features.helpers import get_base_loc, scale
import numpy as np


def make(type_f, source_file, root, output_directory):
    for i, l in enumerate(open(source_file)):
        parts = l.strip().split()
        filename = ' '.join(parts[:-2])
        ref = parts[-2]
        sub = parts[-1]
        h5 = h5py.File(root + "/" + filename, "r")

        def t_to_b(model_state):
            return model_state

        if sub == "1":
            def t_to_b(model_state):
                return model_state.replace("T", "B")

        fo = open(os.path.join(output_directory, "%s.txt" % i), "w")
        fo.writelines(t_to_b(ref))
        base_loc = get_base_loc(h5)

        if type_f in ['temp', 'comp']:
            # scale, scale_sd, shift, drift = extract_scaling(h5, "template", base_loc)
            if type_f == "temp":
                events = h5[base_loc + "/BaseCalled_%s/Events" % "template"]
            else:
                events = h5[base_loc + "/BaseCalled_%s/Events" % "complement"]

            index = 0.0
            data = []

            # events = events[50:-50]
            mean = events["mean"]
            std = events["stdv"]
            length = events["length"]
            X = scale(np.array(np.vstack([mean, mean * mean, std, length]).T, dtype=np.float32))
            g = 0
            for e, (mean, meansqr, std, length) in zip(events, X):
                g += 1
                fo.write(" ".join(map(str, [mean, meansqr, std, length])))
                move = e["move"]
                state = str(e["model_state"])
                if move == 0:
                    fo.write(" NN" + "\n")
                if move == 1:
                    fo.write(" N%s" % t_to_b(state[2]) + "\n")
                if move == 2:
                    fo.write(" %s%s" % (t_to_b(state[1]),
                                        t_to_b(state[2])) + "\n")
                if move in [3, 4, 5]:
                    fo.write(" %s%s" % (t_to_b(state[1]),
                                        t_to_b(state[2])) + "\n")
                if move not in [0, 1, 2]:
                    print("Problem move value =", move, e["model_state"], g, i)
                    print(filename)
                    # exit()

        if type_f == '2d':
            tscale, tscale_sd, tshift, tdrift = extract_scaling(h5, "template", base_loc)
            cscale, cscale_sd, cshift, cdrift = extract_scaling(h5, "complement", base_loc)
            al = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Alignment"]
            temp_events = h5[base_loc + "/BaseCalled_template/Events"]
            comp_events = h5[base_loc + "/BaseCalled_complement/Events"]
            prev = None
            for a in al:
                ev = []
                if a[0] == -1:
                    ev += [0, 0, 0, 0, 0]
                else:
                    e = temp_events[a[0]]
                    mean = (e["mean"] - tshift) / cscale
                    stdv = e["stdv"] / tscale_sd
                    length = e["length"]
                    ev += [1] + preproc_event(mean, stdv, length)
                if a[1] == -1:
                    ev += [0, 0, 0, 0, 0]
                else:
                    e = comp_events[a[1]]
                    mean = (e["mean"] - cshift) / cscale
                    stdv = e["stdv"] / cscale_sd
                    length = e["length"]
                    ev += [1] + preproc_event(mean, stdv, length)
                print >>fo, " ".join(map(str, ev)),
                if prev == a[2]:
                    print >>fo, "NN"
                elif not prev or a[2][:-1] == prev[1:]:
                    print >>fo, "N%c" % a[2][2]
                else:
                    print >>fo, "%c%c" % (a[2][1], a[2][2])

        fo.close()
        h5.close()
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['temp', 'comp', '2d'])
    parser.add_argument('source_file', type=str)
    parser.add_argument('root', type=str)
    parser.add_argument('output_directory', type=str)
    args = parser.parse_args()

    make(type_f=args.type_f, source_file=args.source_file,
         root=args.root, output_directory=args.output_directory)
