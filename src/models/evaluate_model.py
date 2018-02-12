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
    parser.add_argument('--name', type=str)

    parser.add_argument('--weights', dest='weights', type=str, default=None)

    parser.add_argument('--n-input', dest="n_input", type=int, default=1)
    parser.add_argument('--n-output', dest="n_output", type=int, default=1)
    parser.add_argument('--n-output-network', dest="n_output_network", type=int, default=1)
    parser.add_argument('--force-clean', dest="force_clean", action="store_true")
    parser.add_argument('--filter', nargs='+', dest="filter", type=str, default=[])
    parser.add_argument('--ctc-length', dest="ctc_length", type=int, default=20)

    parser.add_argument('--clean', dest="clean", action="store_true")
    parser.add_argument('--sclean', dest="sclean", action="store_true")
    parser.add_argument('--attention', dest="attention", action="store_true")
    parser.add_argument('--residual', dest="res", action="store_true")
    parser.add_argument('--all-datasets', nargs='+', dest="all_datasets", default=[], type=str)

    parser.add_argument('--simple', dest="simple", action="store_true")

    parser.add_argument('--correct-ref', dest="correct_ref", action="store_true")
    parser.add_argument('--num-threads', dest="num_threads", type=int, default=1)

    parser.add_argument('--supcorre', dest="supcorre", action='store_true')
    parser.add_argument('--norm2', dest="norm2", action="store_true")

    parser.add_argument('--substitution', dest="substitution", action="store_true")
    parser.add_argument('--maxf', dest="maxf", type=int, default=None)
    parser.add_argument('--maxlen', dest="maxlen", type=int, default=None)
    parser.add_argument('--maxlen_input', dest="maxlen_input", type=int, default=None)
    parser.add_argument('--target', dest='target', type=str, default=None)

    args = parser.parse_args()

    argparse_dict = vars(args)

    # sess = tf.Session(config=tf.ConfigProto(
#        intra_op_parallelism_threads=args.num_threads))

    repo = Repo("./")
    argparse_dict["commit"] = str(repo.head.commit)

    root, weight = os.path.split(args.weights)

    root += "/evaluate_%s/" % weight.split("-")[-1][:-3]

    root += args.target

    if args.all_datasets == []:
        if args.target == "T":
            args.all_datasets = [
                "/data/bioinfo@borvo/users/jarbona/deepnano5bases/set-T-dataset2-w8-second-half/dataset.pick"]
        if args.target == "B":
            args.all_datasets = [
                "/data/bioinfo@borvo/users/jarbona/deepnano5bases/set-B-dataset2-w5-second-half/dataset.pick"]
        if args.target == "H_B":
            args.all_datasets = [
                "/data/bioinfo@borvo/users/jarbona/deepnano5bases/human_B/dataset.pick"]

    print(root)

    os.makedirs(root, exist_ok=True)

    with open(root + '/params.json', 'w') as fp:
        json.dump(argparse_dict, fp, indent=True)
    # print(args.filter)

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

    from .model import build_models
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

    predictor, _ = build_models(args.size, nbase=args.Nbases - 4,
                                ctc_length=ctc_length,
                                input_length=None, n_output=n_output_network,
                                lr=1, res=args.res, attention=args.attention, n_feat=n_feat, simple=args.simple)

    if args.weights is not None:

        predictor.load_weights(args.weights)
        # except:
        #    print("Learning from scratch")

    original = []
    convert = []

    data_index = []
    data_alignment = []
    refs = []
    names = []

    from ..data.dataset import Dataset
    sys.path.append("src/")

    from ..features.helpers import scale_simple, scale_named, scale_named2

    def load_datasets(argdatasets):

        with open(argdatasets, "rb") as fich:
            D = cPickle.load(fich)

        fnorm = scale_named
        if args.norm2:
            fnorm = scale_named2

        data_x = []

        for strand in D.strands[:args.maxf]:

            if strand.transfered is None:
                data_x.append(None)
                continue

            if args.sclean:
                data_x.append(scale_simple(strand.transfered))
            else:
                data_x.append(fnorm(strand.transfered))
            # if args.raw:
            #     sl = s.sampling_rate
            #     data_x[-1] = [s.raw[int(start * sl):int((start + length) * sl)] for start,
            # length in zip(strand.transfered["start"], strand.transfered["length"])]

        return data_x, D

    for dataset in args.all_datasets:

        data_x, load_dataset = load_datasets(dataset)

        Evaluated_dataset = Dataset("", "", metadata={"original_dataset": load_dataset.metadata,
                                                      "weight_parameter": args.weights,
                                                      "alloptions": argparse_dict})
        Evaluated_dataset.strands = []

        for s, sig in zip(load_dataset.strands[:args.maxf], data_x):
            # s.segmentation(w=8,prefix="../../")
            # transfered = s.transfer(s.signal_bc,s.segments)
            # l = "I0013833_20170821_FAH14319_MN17490_sequencing_run_780_80823_read_344_ch_1_strand"
            # if not (l in s.filename):
            #    continue
            if s.transfered is None:
                continue

            transfered = s.transfered[:args.maxlen_input]

            # sig = np.array(scale_clean_two_pd(transfered), dtype=np.float32)

            # print(sig.shape)
            ns = s.analyse_segmentation(ntwk=predictor, signal=sig[
                                        :args.maxlen_input], no2=n_output_network == 2)

            new = copy.deepcopy(transfered)
            # print(ns.shape)
            new["seq"] = ns[::, 0]

            if n_output_network == 2:

                new["seq1"] = ns[::, 1]

                print(np.sum(new["seq"] == "B"),)
                print(len("".join(transfered["seq"]).replace("N", "")), len("".join(new["seq"]).replace("N", "")),
                      len("".join(new["seq1"]).replace("N", "")), len(new))
            else:
                print(np.sum(new["seq"] == "B"),)
                if "seq" in transfered.columns:
                    print(len("".join(transfered["seq"]).replace("N", "")), len("".join(new["seq"]).replace("N", "")),
                          len(new))
                else:
                    print(len("".join(new["seq"]).replace("N", "")),
                          len(new))

            s.ntwk_align = new

            if n_output_network == 2:
                seq = ""
                for b1, b2 in zip(s.ntwk_align["seq"], s.ntwk_align["seq1"]):
                    seq += b1
                    seq += b2
                print(np.sum(s.ntwk_align["seq"] != "N"), np.sum(s.ntwk_align["seq1"] != "N"))
                seq = "".join(seq).replace("N", "").replace("B", "T")
            else:
                seq = "".join(s.ntwk_align["seq"]).replace("N", "").replace("B", "T")

            s.ref_ntwk = s.get_ref(seq, pos=True, correct=True)
            s.score_ntwk = s.score(seq, s.ref_ntwk[0], maxlen=args.maxlen)
            # print(Score)
            # Score["ntwk_bc"].append(s.score("".join(s.ntwk_align["seq"]).replace("N",""),s.seq_from_basecall))
            if hasattr(s, "seq_from_basecall"):
                s.ref_bc = s.get_ref(s.seq_from_basecall, pos=True, correct=True)

                s.score_mininion = s.score(s.seq_from_basecall, s.ref_bc[0], maxlen=args.maxlen)

            Evaluated_dataset.strands.append(s)

    print(root + "/" + args.target + "-dataset" + ".pick")
    with open(root + "/" + args.target + "-dataset" + ".pick", "wb") as f:
        cPickle.dump(Evaluated_dataset, f)
