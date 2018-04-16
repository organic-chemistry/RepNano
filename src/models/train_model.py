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
import copy
from ..features.helpers import scale_clean, scale_clean_two
from .helper import lrd
import csv
import keras.backend as K


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


import pysam


def rebuild_alignemnt_from_bam(ref, filename="./tmp.sam", debug=False):

    samf = pysam.AlignmentFile(filename)

    read = None
    for read in samf:
        break

    if read is None or read.flag == 4:
        return "", "", False, None, None, None

    # print(read.flag)
    s = read.get_reference_sequence()
    to_match = ref

    # return s,to_match

    if read.is_reverse:
        revcompl = lambda x: ''.join(
            [{'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}[B.upper()] for B in x][::-1])
        to_match = revcompl(ref)
        # print("reverse")

    # print(s[:100])
    # print(to_match[:100])

    to_build = []
    seq_tobuild = []
    index = 0
    for p in read.get_aligned_pairs(with_seq=False, matches_only=False):
        x0, y0 = p

        if x0 is not None:
            x0 = read.seq[x0]
        else:
            x0 = "-"
        if y0 is not None:
            y01 = s[y0 - read.reference_start]
            if index >= len(to_match):
                print("Break")
                break
            if y01.islower() or y01.upper() == to_match[index]:
                # same or mismatch
                to_build.append(y01.upper())
                index += 1

            else:
                to_build.append("-")
        else:
            to_build.append("-")

        y02 = to_build[-1]
        if debug:
            if y0 is None:
                print(x0, y0)
            else:
                print(x0, y02, y01)
        seq_tobuild.append(x0)

    if read.is_reverse:
        seq_tobuild = seq_tobuild[::-1]
        to_build = to_build[::-1]

    start = 0
    for iss, s in enumerate(to_build):
        if s != "-":
            start = iss
            break
    end = None
    for iss, s in enumerate(to_build[::-1]):
        if s != "-":
            end = -iss
            break
    if end == 0:
        end = None
    if read.is_reverse:
        compl = lambda x: ''.join(
            [{'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', "-": "-"}[B.upper()] for B in x])
        return "".join(compl(seq_tobuild)), "".join(compl(to_build)), True, start, end, True
    return "".join(seq_tobuild), "".join(to_build), True, start, end, False


def get_al(se0, ref, tmpfile="./tmp.sam", check=False):
    seq, match, success, start, end, reverse = rebuild_alignemnt_from_bam(ref, tmpfile, debug=False)

    iend = copy.deepcopy(end)
    istart = copy.deepcopy(start)
    endp = 0
    if not reverse:
        if len(se0) != len(seq.replace("-", "")):
            endp = - (len(se0) - len(seq.replace("-", "")))

        if end is None:
            end = 0
        end = end + endp
        if end == 0:
            end = None
    if reverse:
        if len(se0) != len(seq.replace("-", "")):
            endp = (len(se0) - len(seq.replace("-", "")))
        start += endp

    if check:
        print("Checking ref,build_ref")

        for i in range(len(ref) // 100):
            print(ref[i * 100:(i + 1) * 100])
            print(match.replace("-", "")[i * 100:(i + 1) * 100])

        print("Checking seq,build_seq")
        for i in range(len(ref) // 100 + 1):
            print(se0[i * 100:(i + 1) * 100])
            print(seq.replace("-", "")[i * 100:(i + 1) * 100])

    # return where to truncate se0 and the allignement over this portion:
    # seq correspond to se0
    # and match to the ref
    return start, end, seq[istart:iend], match[istart:iend], success

if __name__ == '__main__':

    import argparse
    import json
    from git import Repo
    import tensorflow as tf

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
    parser.add_argument('--ctc', dest='ctc', action="store_true")
    parser.add_argument('--n-input', dest="n_input", type=int, default=1)
    parser.add_argument('--n-output', dest="n_output", type=int, default=1)
    parser.add_argument('--n-output-network', dest="n_output_network", type=int, default=1)
    parser.add_argument('--force-clean', dest="force_clean", action="store_true")
    parser.add_argument('--filter', nargs='+', dest="filter", type=str, default=[])
    parser.add_argument('--ctc-length', dest="ctc_length", type=int, default=20)
    parser.add_argument('--lr', dest="lr", type=float, default=0.01)
    parser.add_argument('--clean', dest="clean", action="store_true")
    parser.add_argument('--sclean', dest="sclean", action="store_true")
    parser.add_argument('--attention', dest="attention", action="store_true")
    parser.add_argument('--residual', dest="res", action="store_true")
    parser.add_argument('--all-datasets', nargs='+', dest="all_datasets", default=[], type=str)
    parser.add_argument('--all-test-datasets', nargs='+',
                        dest="all_test_datasets", default=[], type=str)

    parser.add_argument('--simple', dest="simple", action="store_true")
    parser.add_argument('--all-T', dest="all_T", action="store_true")
    parser.add_argument('--hybrid', dest="hybrid", action="store_true")
    parser.add_argument('--correct-ref', dest="correct_ref", action="store_true")
    parser.add_argument('--all-quality', dest="all_quality", type=float, default=0.)
    parser.add_argument('--num-threads', dest="num_threads", type=int, default=1)
    parser.add_argument('--batch-size', dest="batch_size", type=int, default=10)
    parser.add_argument('--supcorre', dest="supcorre", action='store_true')
    parser.add_argument('--waiting-time', dest="waiting_time", type=int, default=500)
    parser.add_argument('--norm2', dest="norm2", action="store_true")
    parser.add_argument('--norm3', dest="norm3", action="store_true")
    parser.add_argument('--allinfos', dest="allinfos", action="store_true")
    parser.add_argument('--maxleninf', dest="maxleninf", type=int, default=36)
    parser.add_argument('--raw', dest="raw", action="store_true")
    parser.add_argument('--substitution', dest="substitution", action="store_true")
    parser.add_argument('--maxf', dest="maxf", type=int, default=None)
    parser.add_argument('--extra-output', dest='extra_output', type=int, default=0)
    parser.add_argument('--probas', nargs='+', dest="probas", default=[], type=str)

    args = parser.parse_args()

    argparse_dict = vars(args)

    # sess = tf.Session(config=tf.ConfigProto(
#        intra_op_parallelism_threads=args.num_threads))

    repo = Repo("./")
    argparse_dict["commit"] = str(repo.head.commit)

    os.makedirs(args.root, exist_ok=True)

    with open(args.root + '/params.json', 'w') as fp:
        json.dump(argparse_dict, fp, indent=True)
    # print(args.filter)

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

    subseq_size = args.ctc_length

    from .model import build_models, load_weights_from_hdf5_group_what_you_can
    ctc_length = subseq_size
    input_length = ctc_length
    if n_output_network == 2:
        input_length = subseq_size
        ctc_length = 2 * subseq_size

    n_feat = 4
    # if args.clean:
#        n_feat = 3

    if args.sclean:
        n_feat = 1
    if args.norm2:
        n_feat = 2
    if args.norm3:
        n_feat = 3

    if args.allinfos:
        n_feat = args.maxleninf

    if len(args.probas) != args.extra_output:
        print("Pproba must match extra output")
        raise

    _, ntwk = build_models(args.size, nbase=args.Nbases - 4,
                           ctc_length=ctc_length,
                           input_length=input_length, n_output=n_output_network,
                           lr=args.lr, res=args.res,
                           attention=args.attention,
                           n_feat=n_feat, simple=args.simple, extra_output=args.extra_output)
    predictor, _ = build_models(args.size, nbase=args.Nbases - 4,
                                ctc_length=ctc_length,
                                input_length=None, n_output=n_output_network,
                                lr=args.lr, res=args.res, attention=args.attention,
                                n_feat=n_feat, simple=args.simple, extra_output=args.extra_output)

    if args.pre_trained_weight is not None:

        try:
            # try:
            ntwk.load_weights(args.pre_trained_weight)
            # except:
            # print("Only predictor loaded (normal if no ctc)")
            predictor.load_weights(args.pre_trained_weight)
        except:
            load_weights_from_hdf5_group_what_you_can(args.pre_trained_weight, ntwk.layers)
            load_weights_from_hdf5_group_what_you_can(args.pre_trained_weight, predictor.layers)
        # except:
        #    print("Learning from scratch")

    os.makedirs(args.root, exist_ok=True)

    original = []
    convert = []

    data_index = []
    data_alignment = []
    refs = []
    names = []

    from ..data.dataset import Dataset
    from .. import data
    from ..data import dataset
    import sys
    sys.path.append("src/")

    from ..features.helpers import scale_simple, scale_named, scale_named2, scale_named3, scale_named4
    root = "data/raw/20170908-R9.5/"

    def load_datasets(argdatasets):
        Datasets = []
        for d in argdatasets:
            with open(d, "rb") as fich:
                Datasets.append(cPickle.load(fich))

        fnorm = scale_named
        if args.norm2:
            fnorm = scale_named2
        if args.norm3:
            fnorm = scale_named3

        if args.allinfos:
            fnorm = lambda x: scale_named4(x, maxleninf=args.maxleninf)

        data_x = []
        data_y = []
        data_y2 = []
        probas = []

        for D, named in zip(Datasets, argdatasets):
            keep = 0

            for strand in D.strands[:args.maxf]:

                if strand.transfered is None:
                    continue

                if not(strand.bc_score > args.all_quality):
                    continue

                if args.sclean:
                    data_x.append(scale_simple(strand.transfered))
                else:
                    data_x.append(fnorm(strand.transfered))
                if args.raw:
                    sl = s.sampling_rate
                    data_x[-1] = [s.raw[int(start * sl):int((start + length) * sl)] for start,
                                  length in zip(strand.transfered["start"], strand.transfered["length"])]

                def transform(b):

                    if args.Nbases == 4 and b not in ["N", "A", "T", "C", "G"]:
                        return "T"

                    if b != "T":
                        return b

                    if not args.substitution:
                        return "T"
                    if hasattr(D, "substitution") and D.substitution != "T":
                        return D.substitution

                    return "T"

                if args.correct_ref:
                    data_y.append([[mapping[transform(b[0])],
                                    mapping[transform(b2[0])],
                                    mapping[transform(b[1])],
                                    mapping[transform(b2[1])]]
                                   for b, b2 in zip(strand.transfered["seq_ref"],
                                                    strand.transfered["seq_ref_correction"])])

                else:
                    data_y.append([mapping[transform(b)] for b in strand.transfered["seq"]])

                probas.append([])
                for sub in args.probas:
                    if hasattr(strand, "%sprop" % sub):
                        probas[-1].append(getattr(strand, "%sprop" % sub))
                    else:
                        probas[-1].append(0)
                keep += 1
            print(named, len(D.strands[:args.maxf]), keep)
        del Datasets
        return data_x, data_y, data_y2, np.array(probas)

    data_x, data_y, data_y2, probas = load_datasets(args.all_datasets)
    if args.all_test_datasets != []:
        tdata_x, tdata_y, tdata_y2, tprobas = load_datasets(args.all_test_datasets)
    else:
        tdata_x, tdata_y, tdata_y2, tprobas = data_x, data_y, data_y2, probas

    print("done", len(data_x), len(tdata_x))
    # exit()
    sys.stdout.flush()
    # rint(data_x, data_x[0].shape)
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

    ts_arr = []
    tp_arr = []
    for s in range(len(tdata_x)):
        ts_arr += [s]
        tp_arr += [len(tdata_x[s]) - subseq_size]

    tsum_p = sum(tp_arr)
    for i in range(len(tp_arr)):
        tp_arr[i] = 1. * tp_arr[i] / tsum_p

    batch_size = 1
    n_batches = len(data_x) / batch_size
    # print(len(data_x), batch_size, n_batches, datetime.datetime.now())

    boring = False


# ntwk.load_weights("./my_model_weights.h5")
    Schedul = lrd(waiting_time=args.waiting_time, start_lr=args.lr, min_lr=0.0001, factor=2)
    for epoch in range(20000):

        # Test to see if realignment is interesting:

        if False and args.ctc and epoch % 3000 == 0 and (epoch != 0 or args.force_clean):
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
                type_sub = "T"
                subts = False
                ref = "" + refs[s]

                for l in ["B", "L", "E", "I"]:
                    if l in refs[s]:
                        type_sub = l
                        subts = True
                        break
                if subts:
                    ref = ref.replace(type_sub, "T")

                re_align = True

                if re_align:
                    old_align = data_alignment[s]
                    # new_align = pairwise2.align.globalxx(ref, New_seq[s].replace("N", ""))[0][:2]
                    try:
                        new_align = pairwise2.align.globalxx(ref, New_seq[s].replace("N", ""))
                    except MemoryError:
                        print("Out of memory")
                        continue
                    if len(new_align) == 0 or len(new_align[0]) < 2:
                        new_length += len(old_align[0])
                        print()
                        continue
                    new_align = new_align[0][:2]
                    print("Old", len(old_align[0]), "New", len(new_align[0]), subts, len(
                        ref), (len(ref) - len(New_seq[s].replace("N", ""))) / len(ref), nc[type_sub] / (nc["T"] + 1))

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

                if subts and nc[type_sub] / (nc["T"] + nc[type_sub]) < 0.1:
                    if args.force_clean and type_sub != "B":
                        continue
                    refs[s] = refs[s].replace(type_sub, "T")
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

        def get_transformed_sets(d_x, d_y, d_y2, d_prob, s_arr, p_arr, mini=200, maxi=None):

            print(len(d_x), len(d_y))

            X_new = []
            Y_new = []
            Y2_new = []
            Label = []
            Length = []
            Probas = []
            stats = defaultdict(int)
            megas = ""
            infostat = {}
            stop = False
            while len(X_new) < mini:
                print(len(X_new))
                if stop:
                    break
                for s in range(len(d_x)):
                    if maxi is not None:
                        if len(X_new) >= maxi:
                            stop = True
                            break
                    s2 = np.random.choice(s_arr, p=p_arr)
                    # print(s2)
                    # print(data_x[s2].shape[0])
                    r = np.random.randint(0, d_x[s2].shape[0] - subseq_size)
                    x = d_x[s2][r:r + subseq_size]

                    if not args.ctc:

                        def domap(base):
                            ret = [0 for b in range(args.Nbases + 1)]
                            ret[base] = 1
                            return ret

                        y = [domap(base) for base in d_y[s2][r: r + subseq_size]]
                        y2 = [domap(base) for base in d_y2[s2][r: r + subseq_size]]

                        X_new.append(x)
                        Y_new.append(y)
                        Y2_new.append(y2)

                        for xx in d_y[s2][r:r + subseq_size]:
                            stats[xx] += 1

                    if args.ctc:
                        if not args.correct_ref:
                            y = [base for base in d_y[s2][
                                r: r + subseq_size] if base != mapping["N"]]
                        if args.correct_ref:
                            y = []
                            for b1 in d_y[s2][r: r + subseq_size]:
                                for bb in b1:
                                    if bb != mapping["N"]:
                                        y.append(bb)

                        if y == [] or len(y) > subseq_size:
                            continue

                        X_new.append(x)
                        Label.append(y + [0] * (subseq_size * args.n_output_network - len(y)))
                        Length.append(len(y))
                        Probas.append(d_prob[s2])

                    # x[:,0] += np.random.binomial(n=1, p=0.1, size=x.shape[0]) *
                    # np.random.normal(scale=0.01, size=x.shape[0])

                    # oversampleb
                    # if "B" not in refs[s2] and np.random.randint(args.oversampleb) != 0:
                #        continue

                    if args.ctc and False:
                        def domap(base):
                            ret = [0 for b in range(n_classes)]
                            ret[base] = 1
                            return ret

                        length = subseq_size
                        start = r
                        Index = d_index[s2]
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
                            # print(ss2, ss1, delta, len(ss2.replace("-", "")))
                            # print("Skip")
                            continue
                        print("Keep", delta, ss2, ss1, len(data_x), [
                            l in refs[s2] for l in ["B", "L", "E", "I"]])
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

                        for l in ["T", "B", "L", "E", "I"]:
                            if l in seg:
                                infostat[l] = infostat.get(l, 0) + seg.count(l)

                        seg = seg + "A" * (maxi - len(seg))

                        if not args.all_T:
                            for l in ["B", "L", "E", "I"]:
                                if l in refs[s2]:
                                    if not args.hybrid:
                                        seg = seg.replace("T", l)
                                    break

                        # print(ss1, ss2, seg)

                        # print(len(s))
                        # print(s)
                        # print([base for base in s])
                        Label.append([mapping[base] for base in seg])

                        X_new.append(x)
                        # print(x)

            X_new = np.array(X_new)
            Y_new = np.array(Y_new)
            Y2_new = np.array(Y2_new)
            Label = np.array(Label)
            Length = np.array(Length)

            return X_new, Y_new, Y2_new, Label, Length, stats, Probas

        X_new, Y_new, Y2_new, Label, Length, stats, sp1 = get_transformed_sets(
            data_x, data_y, data_y2, probas, s_arr, p_arr, mini=200)
        tX_new, tY_new, tY2_new, tLabel, tLength, stats, stp1 = get_transformed_sets(
            tdata_x, tdata_y, tdata_y2, tprobas, ts_arr, tp_arr, maxi=40)

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
            if args.n_output_network == 1:
                r = predictor.fit(X_new, Y_new, nb_epoch=1,
                                  batch_size=args.batch_size, validation_split=0.05)
            else:
                r = predictor.fit(X_new, [Y_new, Y2_new], nb_epoch=1,
                                  batch_size=args.batch_size, validation_split=0.05)
            # ntwk.fit(X_new, Y_new, nb_epoch=1, batch_size=10, validation_split=0.05,
            #          sample_weight={"out_layer2": w2}, )
            if epoch % 10 == 0:
                predictor.save_weights(os.path.join(
                    args.root, 'my_model_weights-%i.h5' % epoch))

        if args.ctc:
            # print(megas.count("B") / len(megas), megas.count("T") / len(megas))

            print(X_new.shape)
            print(tLabel.shape, np.array([subseq_size] *
                                         len(tLength)).shape, tLength.shape, tLabel.shape)
            print(Label.shape, np.array([subseq_size] *
                                        len(Length)).shape, Length.shape, Label.shape)

            print(X_new.dtype, Y_new.dtype, Label.dtype, Length.dtype)

            # To balance class weight

            # print(Label)
            # print(X_new.shape, Label.shape, np.array(
            #    [length] * len(Length)).shape, Length.shape)

            if args.test:
                maxin = 8
                val = 2
                batch_size = 8
            else:
                maxin = args.batch_size * (int(len(X_new) // args.batch_size))
                val = 30
                batch_size = args.batch_size

            if args.extra_output:

                def proportion(x, c):
                    return np.sum(x == c, axis=-1) / (np.sum(x == c, axis=-1) + np.sum(x == 3, axis=-1) + 1e-7)

                def countT(x):
                    return np.sum(x == mapping["T"], axis=-1)

                if args.extra_output == 1:
                    p1 = countT(Label)[::, np.newaxis] * sp1 / 1.0 / subseq_size
                    tp1 = countT(tLabel)[::, np.newaxis] * stp1 / 1.0 / subseq_size

                    print(countT(Label))
                    print(sp1)
                    print(p1)

                    r = ntwk.fit([X_new[:maxin], Label[:maxin], np.array([subseq_size] * len(Length))[:maxin], Length[:maxin]],
                                 [Label[:maxin]] + [pi[:maxin] for pi in p1.T], nb_epoch=1, batch_size=batch_size,
                                 validation_data=([tX_new,
                                                   tLabel,
                                                   np.array([subseq_size] *
                                                            len(tLength)),
                                                   tLength],
                                                  [tLabel] + [pi[:maxin] for pi in tp1.T]))

            else:
                r = ntwk.fit([X_new[:maxin], Label[:maxin], np.array([subseq_size] * len(Length))[:maxin], Length[:maxin]],
                             Label[:maxin], nb_epoch=1, batch_size=batch_size,
                             validation_data=([tX_new,
                                               tLabel,
                                               np.array([subseq_size] *
                                                        len(tLength)),
                                               tLength],
                                              tLabel))
            if epoch % 10 == 0:
                ntwk.save_weights(os.path.join(
                    args.root, 'my_model_weights-%i.h5' % epoch))

        csv_keys = ["epoch", "loss", "val_loss"]
        if args.extra_output >= 1:
            csv_keys.extend(["val_ctc_loss", "ctc_loss"])

            for i in range(args.extra_output):
                csv_keys.extend(["o%i_loss" % i, "val_o%i_loss" % i])

        lr = Schedul.set_new_lr(r.history["loss"][0])

        K.set_value(ntwk.optimizer.lr, lr)
        K.set_value(predictor.optimizer.lr, lr)

        print(lr)

        if epoch == 0:
            with open(os.path.join(args.root, "training.log"), "w") as csv_file:
                writer = csv.writer(csv_file)
                # from IPython import embed
                # embed()
                # print(r)
                writer.writerow(csv_keys + ["lr"])
                writer.writerow([epoch] + [r.history[k][-1] for k in csv_keys[1:]] + [lr])
        else:
            with open(os.path.join(args.root, "training.log"), "a") as csv_file:
                writer = csv.writer(csv_file)
                # writer.writerow(k + ["lr"])
                writer.writerow([epoch] + [r.history[k][-1] for k in csv_keys[1:]] + [lr])
        if Schedul.stop:
            exit()

        # if epoch == 0:
        #    ntwk.fit(X_new,[Y_new,Y2_new],nb_epoch=1, batch_size=10,validation_split=0.05)

#  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
