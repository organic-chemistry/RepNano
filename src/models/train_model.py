import sys
import numpy as np
import datetime
from collections import defaultdict
import os
# from sklearn.metrics import confusion_matrix
import glob
import keras
from Bio import pairwise2
import _pickle as cPickle


def print_stats(o):
    stats = defaultdict(int)
    for x in o:
        stats[x] += 1
    print(stats)


def flatten2(x):
    return x.reshape((x.shape[0] * x.shape[1], -1))


def find_closest(start, Index, factor=3.5):
    # Return the first element != N which correspond to the index of seqs
    start_index = min(int(start / factor), len(Index) - 1)
    # print(start,start_index,Index[start_index])
    if Index[start_index] >= start:
        while start_index >= 0 and Index[start_index] >= start:
            start_index -= 1
        return max(0, start_index)

    if Index[start_index] < start:
        while start_index <= len(Index) - 1 and Index[start_index] < start:
            start_index += 1
        if start_index <= len(Index) - 1 and start_index > 0:
            if abs(Index[start_index] - start) > abs(Index[start_index - 1] - start):
                start_index -= 1

            # print(start_index,Index[start_index])
        # print(start_index,min(start_index,len(Index)-1),Index[min(start_index,len(Index)-1)])
        return min(start_index, len(Index) - 1)


def get_segment(alignment, start_index_on_seqs, end_index_on_seqs):
    s1, s2 = alignment
    count = 0
    # print(s1,s2)
    startf = False
    end = None
    # found_end =
    for N, (c1, c2) in enumerate(zip(s1, s2)):
        # print(count)
        if count == start_index_on_seqs and not startf:
            start = 0 + N
            startf = True

        if count == end_index_on_seqs + 1:
            end = 0 + N
            break

        if c2 != "-":
            count += 1

    # print(start,end)
    if not startf:
        return "", "", "", 0
    return s1[start:end].replace("-", ""), s1[start:end], s2[start:end], 1


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--Nbases', type=int, choices=[4, 5, 8], default=4)
    parser.add_argument('--root', type=str, default="data/training/")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--size', type=int, default=20)

    parser.add_argument('directories', type=str, nargs='*')

    parser.add_argument('--from-pre-trained', dest='from_pre_trained', action='store_true')
    parser.add_argument('--pre-trained-weight', dest='pre_trained_weight', type=str, default=None)
    parser.add_argument('--pre-trained-dir-list', dest='pre_trained_dir_list', type=str)
    parser.add_argument('--deltaseq', dest='deltaseq', type=int, default=10)
    parser.add_argument('--forcelength', dest='forcelength', type=float, default=0.5)
    parser.add_argument('--oversampleb', dest='oversampleb', type=int, default=3)
    parser.add_argument('--ref-from-file', dest="ref_from_file", type=bool, default=False)
    parser.add_argument('--select-agree', dest="select_agree", type=bool, default=False)
    parser.add_argument('--max-file', dest="max_file", type=int, default=None)
    parser.add_argument('--ctc', dest='ctc', action="store_true")
    parser.add_argument('--convert-to-t', dest='convert_to_t', type=float, default=None)
    parser.add_argument('--n-input', dest="n_input", type=int, default=1)
    parser.add_argument('--n-output', dest="n_output", type=int, default=1)
    parser.add_argument('--n-output-network', dest="n_output_network", type=int, default=1)
    parser.add_argument('--f-size', nargs='+', dest="f_size", type=int, default=None)
    parser.add_argument('--skip-new', dest="skip_new", action="store_true")

    args = parser.parse_args()

    data_x = []
    data_y = []
    data_y2 = []

    data_index = []
    data_alignment = []
    refs = []
    names = []

    log_total_length = os.path.join(args.root, "total_length.log")
    if keras.backend.backend() != 'tensorflow':
        print("Must use tensorflow to train")
        exit()

    if args.Nbases == 4:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}  # Modif
    elif args.Nbases == 5:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B": 4, "N": 5}  # Modif
    elif args.Nbases == 8:
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B": 4, "L": 5, "E": 6, "I": 7, "N": 8}  # Modif

    n_classes = len(mapping.keys())

    n_output_network = args.n_output_network
    n_output = args.n_output
    n_input = args.n_input

    subseq_size = 50

    from .model import build_models
    ctc_length = subseq_size
    input_length = None
    if n_output_network == 2:
        input_length = subseq_size
        ctc_length = 2 * subseq_size

    predictor, ntwk = build_models(args.size, nbase=args.Nbases - 4,
                                   ctc_length=ctc_length, input_length=input_length, n_output=n_output_network)

    if args.Nbases == 8:
        old_predictor, old_ntwk = build_models(args.size, nbase=1)

    os.makedirs(args.root, exist_ok=True)

    if not os.path.exists(os.path.join(args.root, "Allignements-bis")):
        end = None
        if args.test:
            end = 80

        if not args.from_pre_trained:

            list_files = []
            for folder in args.directories:
                fiches = glob.glob(folder + "/*")
                fiches.sort()
                list_files += fiches[:args.max_file]

            list_files.sort()

            for fn in list_files[:end]:
                f = open(fn)
                ref = f.readline()
                ref = ref.replace("\n", "")
                if len(ref) > 30000:
                    print("out", len(ref))
                    continue

                X = []
                Y = []
                seq = []
                for l in f:
                    its = l.strip().split()
                    X.append(list(map(float, its[:-1])))

                    if n_output == 2:
                        Y.append(mapping[its[-1][0]])
                        Y.append(mapping[its[-1][1]])
                    else:
                        Y.append(mapping[its[-1][1]])

                    if n_input == 2:
                        X.append(list(map(float, its[:-1])))

                    seq.append(its[-1])

                if len(X) < subseq_size:
                    print("out (too small (to include must set a smaller subseq_size))", fn)
                    continue

                refs.append(ref.strip())
                names.append(fn)
                data_x.append(np.array(X, dtype=np.float32))
                data_y.append(np.array(Y, dtype=np.int32))

                if n_output == 2 and args.convert_to_t:
                    p = np.sum(data_y[-1] == 5) / len(Y)
                    if p > args.convert_to_t:
                        print(np.sum(data_y[-1] == mapping["B"]))
                        data_y[-1][data_y[-1] == mapping["B"]] = mapping["T"]
                        print(np.sum(data_y[-1] == mapping["B"]))
                        print("Converted")

                print(fn, np.sum(data_y[-1] == 5) / len(Y))

                # print(data_y2[-1][:20])
                # print(data_y[-1][:20])

                if args.ctc:

                    on_ref = False
                    if on_ref:
                        seq = "".join(seq)
                        # print(seq)
                        seq = seq[1::2]
                        # print(seq)
                        data_index.append(np.arange(len(seq))[np.array([s for s in seq]) != "N"])
                        seqs = seq.replace("N", "")
                        alignments = pairwise2.align.globalxx(ref, seqs)
                        data_alignment.append(alignments[0][:2])
                        # print(len(seqs), len(ref))
                        print(len(alignments[0][0]), len(ref), len(seqs), alignments[0][2:])
                    else:
                        seq = "".join(seq)
                        if n_output == 1:
                            seq = seq[1::2]

                        # print(seq)
                        # print(seq)
                        data_index.append(np.arange(len(seq))[np.array([s for s in seq]) != "N"])
                        seqs = seq.replace("N", "")
                        data_alignment.append([seqs, seqs])

            if not args.ctc:
                with open(os.path.join(args.root, "Allignements-bis"), "wb") as f:
                    cPickle.dump([data_x, data_y, data_y2, refs, names], f)
            else:
                with open(os.path.join(args.root, "Allignements-bis"), "wb") as f:
                    cPickle.dump([data_x, data_index, data_alignment, refs, names], f)

        else:

            if args.Nbases == 5:
                ntwk.load_weights(args.pre_trained_weight)
                predictor.load_weights(args.pre_trained_weight)
            else:
                old_predictor.load_weights(args.pre_trained_weight)

            from ..features.extract_events import extract_events, scale
            import h5py
            import subprocess
            from ..features.bwa_tools import get_seq
            end = None
            if args.test:
                end = 10

            with open(args.pre_trained_dir_list, "r") as f:
                idirect = 0
                for iline, line in enumerate(f.readlines()):
                    if not args.ref_from_file:
                        if len(line.split()) not in [2, 3]:
                            print("Skipping ", line)
                            continue
                        if len(line.split()) == 2:
                            direct, type_sub = line.split()
                        else:
                            direct, type_sub, ref_file = line.split()
                    else:
                        if len(line.split()) != 3:
                            print("Skipping ", line)
                            continue

                        direct, type_sub, ref_file = line.split()
                    idirect += 1
                    sub = None
                    type_sub = type_sub.strip()
                    if type_sub != "T":
                        sub = type_sub
                        if sub not in mapping:
                            raise "Invalid substitution"

                    for ifilename, filename in enumerate(glob.glob(direct + "/*")):
                        if args.max_file is not None and ifilename > args.max_file:
                            continue
                        h5 = h5py.File(filename, "r")

                        if args.f_size is not None:
                            events = extract_events(h5, "rf", window_size=args.f_size[iline])
                        else:
                            events = extract_events(h5, "r9.5")

                        if events is None:
                            print("No events in file %s" % filename)
                            h5.close()
                            continue

                        if len(events) < 300:
                            print("Read %s too short, not basecalling" % filename)
                            h5.close()
                            continue
                        # print(len(events))
                        if args.test and len(events) > 2500:
                            continue
                        if args.test and len(data_x) > (iline + 1) * 10:
                            break
                        events = events[1:-1]
                        mean = events["mean"]
                        std = events["stdv"]
                        length = events["length"]
                        x = scale(
                            np.array(np.vstack([mean, mean * mean, std, length]).T, dtype=np.float32))

                        if args.Nbases == 5:
                            o1 = predictor.predict(np.array(x)[np.newaxis, ::, ::])
                        elif args.Nbases == 8:
                            o1 = old_predictor.predict(np.array(x)[np.newaxis, ::, ::])
                        o1 = o1[0]
                        om = np.argmax(o1, axis=-1)

                        alph = "ACGTN"
                        if args.Nbases == 5:
                            alph = "ACGTTN"
                        if args.Nbases == 8:
                            alph = "ACGTTTTTN"

                        seq = "".join(map(lambda x: alph[x], om))
                        seqs = seq.replace("N", "")

                        # write fasta
                        with open("tmp.fasta", "w") as output_file:
                            output_file.writelines(">%s_template_deepnano\n" % filename)
                            output_file.writelines(seqs + "\n")

                        # execute bwa

                        if not args.ref_from_file or args.select_agree:
                            ref = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"
                            exex = "bwa mem -x ont2d  %s  tmp.fasta > tmp.sam" % ref
                            subprocess.call(exex, shell=True)

                            # read from bwa

                            ref, succes, X1, P1 = get_seq(
                                "tmp.sam", ref="data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa", ret_pos=True)

                            if not succes:
                                continue

                        if args.ref_from_file or args.select_agree:
                            k = filename.split("/")[-1]
                            read, ch = k.split("_")[9], k.split("_")[11]
                            succes = False
                            Ref = []
                            with open(ref_file, "r") as f:
                                for line in f.readlines():
                                    sp = line.split()

                                    if len(sp) > 1 and sp[0].startswith("@ch"):
                                        kp = sp[0].split("/")[-1]

                                        chp = kp.split("_")[0][3:]
                                        readp = kp.split("_")[1][4:]

                                        if read == readp and ch == chp:
                                            print(k, kp)

                                            if sp[2] == '*' or "chr" not in sp[2]:
                                                continue

                                            X2 = int(sp[2][3:])
                                            P2 = int(sp[3])
                                            ref = sp[9]
                                            Ref.append(["" + ref, X2, P2])
                                            succes = True
                                            # break
                            if succes:
                                if not args.select_agree:
                                    ref = list(sorted(Ref, key=lambda x: len(x[0])))[-1][0]
                                    print([len(iRef[0]) for iRef in Ref])

                                    print(len(ref), len(seqs))
                                else:
                                    found = False
                                    for seq2, X2, P2 in Ref:
                                        if X1 == X2 and abs(P1 - P2) < 5000:
                                            found = True
                                            print("Agreee")
                                    if not found:
                                        continue
                            else:
                                continue

                            if abs(len(ref) - len(seqs)) > 1000:
                                succes = False

                            if not succes:
                                continue

                        if args.test:
                            print(len(data_x), "LEN")
                            if len(ref) > 2000 or len(seqs) > 2000:
                                continue
                            if len(data_x) > 20 * idirect:
                                break
                        if len(ref) > 30000:
                            print("out", len(ref))
                            continue
                        if succes:
                            alignments = pairwise2.align.globalxx(
                                ref, seqs, one_alignment_only=True)
                            # print("la", len(alignments), len(alignments[0]))
                            if len(alignments) > 0 and len(alignments[0]) >= 2:

                                names.append(filename)

                                data_x.append(x)
                                data_index.append(np.arange(len(seq))[
                                                  np.array([s for s in seq]) != "N"])

                                data_alignment.append(alignments[0][:2])
                                if sub is not None:
                                    ref = ref.replace("T", sub)
                                # print(ref)
                                refs.append(ref)
                                # print(len(seqs), len(ref))
                                print(len(alignments[0][0]), len(ref), len(seqs), alignments[0][2:])
                        else:
                            print("Fail")

            with open(os.path.join(args.root, "Allignements-bis"), "wb") as f:
                cPickle.dump([data_x, data_index, data_alignment, refs, names], f)
    else:
        if args.ctc:
            with open(os.path.join(args.root, "Allignements-bis"), "rb") as f:
                data_x, data_index, data_alignment, refs, names = cPickle.load(f)
        else:
            with open(os.path.join(args.root, "Allignements-bis"), "rb") as f:
                data_x, data_y, data_y2, refs, names = cPickle.load(f)

    if args.pre_trained_weight is not None:

        try:
            try:
                ntwk.load_weights(args.pre_trained_weight)
            except:
                print("Only predictor loaded (normal if no ctc)")
            predictor.load_weights(args.pre_trained_weight)
        except:
            print("Learning from scratch")

    print("done", sum(len(x) for x in refs))
    sys.stdout.flush()
    # print(len(refs[0]),len(data_x[0]),len(data_y[0]))
    # exit()

    s_arr = []
    p_arr = []
    for s in range(len(data_x)):
        s_arr += [s]
        p_arr += [len(data_x[s]) - subseq_size]

    sum_p = sum(p_arr)
    for i in range(len(p_arr)):
        p_arr[i] = 1. * p_arr[i] / sum_p

    batch_size = 1
    n_batches = len(data_x) / batch_size
    print(len(data_x), batch_size, n_batches, datetime.datetime.now())

    boring = False


# ntwk.load_weights("./my_model_weights.h5")
    for epoch in range(10000):

        # Test to see if realignment is interesting:

        if args.ctc and epoch % 2000 == 0 and epoch != 0 and False:
            ntwk.save_weights(os.path.join(
                args.root, 'tmp.h5'))

            predictor.load_weights(os.path.join(
                args.root, 'tmp.h5'))

            # predictor.load_weights("data/training/my_model_weights-1990.h5")

            print("Realign")
            New_seq = []
            change = 0
            old_length = 0
            new_length = 0
            total_length = 0
            current_length = 0
            switch = 0
            for s in range(len(data_x)):

                new_seq = np.argmax(predictor.predict(np.array([data_x[s]]))[0], axis=-1)
                # print(args.Nbases)
                if args.Nbases == 8:
                    alph = "ACGTBLEIN"   # use T to Align
                if args.Nbases == 5:
                    alph = "ACGTBN"   # use T to Align
                if args.Nbases == 4:
                    alph = "ACGTN"
                New_seq.append("".join(list(map(lambda x: alph[x], new_seq))))

                nc = {}

                for l in ["B", "L", "E", "I", "T"]:
                    nc[l] = New_seq[-1].count(l)

                for l in ["B", "L", "E", "I"]:
                    New_seq[-1] = New_seq[-1].replace(l, "T")

            # Here maybe realign with bwa
            # for s in range(len(data_x)):
                old_align = data_alignment[s]

                type_sub = "T"
                subts = False
                for l in ["B", "L", "E", "I"]:
                    if l in ref[s]:
                        subst = l
                        break
                ref = "" + refs[s]
                if subts:
                    ref = ref.replace(subts, "T")

                # new_align = pairwise2.align.globalxx(ref, New_seq[s].replace("N", ""))[0][:2]
                new_align = pairwise2.align.globalxx(ref, New_seq[s].replace("N", ""))
                if len(new_align) == 0 or len(new_align[0]) < 2:
                    new_length += len(old_align[0])
                    print()
                    continue
                new_align = new_align[0][:2]
                print("Old", len(old_align[0]), "New", len(new_align[0]), subts, len(
                    ref), (len(ref) - len(New_seq[s].replace("N", ""))) / len(ref), nc[subts] / (nc["T"] + 1))

                old_length += len(old_align[0])
                total_length += len(ref)
                current_length += len(New_seq[s].replace("N", ""))
                if len(new_align[0]) < len(old_align[0]) and (len(ref) - len(New_seq[s].replace("N", ""))) / len(ref) < 0.05:
                    print("Keep!")
                    change += 1
                    data_alignment[s] = new_align

                    data_index[s] = np.arange(len(New_seq[s]))[
                        np.array([ss for ss in New_seq[s]]) != "N"]
                    new_length += len(new_align[0])

                else:
                    new_length += len(old_align[0])
                    print()

                if subts and nc[subts] / (nc["T"] + 1) < 0.3:
                    refs[s] = refs[s].replace(subts, "T")
                    switch += 1
                    print("Swich")
            print("Change", change, len(data_x))
            with open(os.path.join(
                    args.root, "Allignements-bis-%i" % epoch), "wb") as f:
                cPickle.dump([data_x, data_index,
                              data_alignment, refs, names], f)
            with open(log_total_length, "a") as f:
                f.writelines("%i,%i,%i,%i,%i,%i,%i\n" %
                             (epoch, old_length, new_length, total_length, current_length, change, switch))

            # Keep new alignment

        taken_gc = []
        out_gc = []
        tc = 0
        tc2 = 0
        tc3 = 0

        X_new = []
        Y_new = []
        Y2_new = []
        Label = []
        Length = []
        stats = defaultdict(int)
        megas = ""
        stats = defaultdict(int)

        while len(X_new) < 200:
            print(len(X_new))
            for s in range(len(data_x)):
                s2 = np.random.choice(s_arr, p=p_arr)
                r = np.random.randint(0, data_x[s2].shape[0] - subseq_size)
                x = data_x[s2][r:r + subseq_size]

                if not args.ctc:

                    def domap(base):
                        ret = [0 for b in range(args.Nbases + 1)]
                        ret[base] = 1
                        return ret

                    y = [domap(base) for base in data_y[s2][r: r + subseq_size]]

                    X_new.append(x)
                    Y_new.append(y)

                    for xx in data_y[s2][r:r + subseq_size]:
                        stats[xx] += 1
                # x[:,0] += np.random.binomial(n=1, p=0.1, size=x.shape[0]) *
                # np.random.normal(scale=0.01, size=x.shape[0])

                # oversampleb
                # if "B" not in refs[s2] and np.random.randint(args.oversampleb) != 0:
            #        continue
                if args.ctc:
                    def domap(base):
                        ret = [0 for b in range(n_classes)]
                        ret[base] = 1
                        return ret
                    skip = False
                    if args.skip_new:
                        for l in ["L", "E", "I"]:
                            if l in refs[s2]:
                                skip = True
                    if skip:
                        continue
                    length = subseq_size
                    start = r
                    Index = data_index[s2]
                    alignment = data_alignment[s2]
                    f = 1
                    if n_input == 1 and n_output == 2:
                        f = 2

                    start_index_on_seqs = find_closest(start * f, Index)
                    end_index_on_seqs = find_closest(start * f + length * f, Index)
                    # from IPython import embed
                    # embed()
                    # print(start, start_index_on_seqs, end_index_on_seqs,
                    #      len(alignment[0]), len(alignment[1]))
                    seg, ss1, ss2, success = get_segment(
                        alignment, start_index_on_seqs, end_index_on_seqs)

                    # print(ss2, ss1, seg, [l in refs[s2] for l in ["B", "L", "E", "I"]])

                    if not success:
                        continue
                    maxi = f * subseq_size
                    l = min(max(len(seg), 1), maxi - 1)
                    delta = abs(len(ss2.replace("-", "")) - len(ss2)) + \
                        abs(len(ss1.replace("-", "")) - len(ss1))

                    if delta > args.deltaseq or \
                            len(ss2.replace("-", "")) < args.forcelength * subseq_size or len(ss1.replace("-", "")) < args.forcelength * subseq_size:
                        #print(ss2, ss1, delta, len(ss2.replace("-", "")))
                        # print("Skip")
                        continue
                    #print("Keep", delta, ss2, ss1)
                    Length.append(l)

                    test = False
                    if test:
                        # print(len(data_x[s2]))
                        o1 = predictor.predict(np.array(x)[np.newaxis, ::, ::])
                        o1 = o1[0]
                        om = np.argmax(o1, axis=-1)

                        alph = "ACGTTN"
                        seq_tmp = "".join(map(lambda x: alph[x], om))
                        print(seq_tmp.replace("N", ""))

                    # print(len(s))
                    if len(seg) > maxi - 1:
                        seg = seg[:maxi - 1]

                    if "B" in refs[s2]:
                        megas += seg.replace("T", "B")
                    else:
                        megas += seg

                    seg = seg + "A" * (maxi - len(seg))
                    for l in ["B", "L", "E", "I"]:
                        if l in refs[s2]:

                            seg = seg.replace("T", l)
                            break

                    # print(ss1, ss2, seg)

                    # print(len(s))
                    # print(s)
                    # print([base for base in s])
                    Label.append([mapping[base] for base in seg])

                    X_new.append(x)

        X_new = np.array(X_new)
        Y_new = np.array(Y_new)

        if not args.ctc:
            sum1 = 0
            for k in stats.keys():
                sum1 += stats[k]

            if epoch == 0:
                weight = [0 for k in stats.keys()]

                for k in stats.keys():
                    weight[k] = stats[k] / 1.0 / sum1
                    weight[k] = 1 / weight[k]
                weight = np.array(weight)
                weight = weight * len(stats.keys()) / np.sum(weight)
            # weight[4] *= 100

            w2 = []

            for y in Y2_new:
                w2.append([])
                for arr in y:
                    w2[-1].append(weight[np.argmax(arr)])

            w2 = np.array(w2)

        if not args.ctc:
            predictor.fit(X_new, Y_new, nb_epoch=1, batch_size=10, validation_split=0.05)
            # ntwk.fit(X_new, Y_new, nb_epoch=1, batch_size=10, validation_split=0.05,
            #          sample_weight={"out_layer2": w2}, )
            if epoch % 10 == 0:
                predictor.save_weights(os.path.join(
                    args.root, 'my_model_weights-%i.h5' % epoch))

        if args.ctc:
            print(megas.count("B") / len(megas), megas.count("T") / len(megas))

            Label = np.array(Label)
            Length = np.array(Length)
            print(X_new.shape)

            # To balance class weight

            # print(Label)
            # print(X_new.shape, Label.shape, np.array(
            #    [length] * len(Length)).shape, Length.shape)

            if args.test:
                maxin = 8
                val = 2
                batch_size = 8
            else:
                maxin = 10 * (int(len(X_new) // 10) - 3)
                val = 30
                batch_size = 10
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                          patience=5, min_lr=0.0001)
            Log = keras.callbacks.CSVLogger(filename=os.path.join(
                args.root, "training.log"), append=True)

            # print(len(data_x), np.mean(Length), np.max(Length))
            ntwk.fit([X_new[:maxin], Label[:maxin], np.array([subseq_size] * len(Length))[:maxin], Length[:maxin]],
                     Label[:maxin], nb_epoch=1, batch_size=batch_size, callbacks=[reduce_lr, Log],
                     validation_data=([X_new[maxin:maxin + val],
                                       Label[maxin:maxin + val],
                                       np.array([subseq_size] *
                                                len(Length))[maxin:maxin + val],
                                       Length[maxin:maxin + val]],
                                      Label[maxin:maxin + val]))

            # if epoch == 0:
            #    ntwk.fit(X_new,[Y_new,Y2_new],nb_epoch=1, batch_size=10,validation_split=0.05)
            if epoch % 10 == 0:
                ntwk.save_weights(os.path.join(
                    args.root, 'my_model_weights-%i.h5' % epoch))

#  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
