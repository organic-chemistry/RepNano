if __name__ == "__main__":
    import argparse
    import json
    from git import Repo
    import os
    from multiprocessing import Pool
    import numpy as np

    from ..data.dataset import Dataset, NotAllign
    from ..features.helpers import scale_simple, scale_named, scale_named2, scale_named4, scale_named4s
    from ..features.extract_events import tv_segment
    import glob
    import os
    import sys
    from keras import backend as K
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', dest="window_size",
                        type=int, choices=[2, 4, 5, 8], default=5)
    parser.add_argument('--Nbases', type=int, choices=[4, 5, 8], default=4)

    parser.add_argument('--root', type=str, default="./")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--size', type=int, default=20)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=None)
    parser.add_argument("--name", dest="name", type=str, default='dataset.pick')
    parser.add_argument("--target", dest="target", type=str, default='T')
    parser.add_argument("--test-set", dest="test_set", action="store_true")
    parser.add_argument("--range", dest="range", nargs='+', default=[], type=float)

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

    parser.add_argument('--num-threads', dest="num_threads", type=int, default=1)

    parser.add_argument('--norm2', dest="norm2", action="store_true")

    parser.add_argument('--maxf', dest="maxf", type=int, default=None)
    parser.add_argument("--method", dest="method", choices=["FW", "TV", "TV45", "TV25", "TV5"])
    parser.add_argument('--allinfos', dest='allinfos', action='store_true')
    parser.add_argument('--maxleninf', dest="maxleninf", type=int, default=36)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--maxlen', dest="maxlen", type=int, default=10000)
    parser.add_argument('--correct', dest='correct', action='store_true')
    parser.add_argument('--gamma', dest="gamma", type=float, default=40)
    parser.add_argument('--not-normed', dest="normed", action="store_false")
    parser.add_argument('--extra-output', dest='extra_output', type=int, default=0)
    parser.add_argument('--info', action="store_true")
    parser.add_argument('--flatten', action="store_true")
    parser.add_argument('--batchnorm', dest='batchnorm', action="store_true")
    parser.add_argument('--dropout', dest='dropout', default=0, type=float)
    parser.add_argument("--human", dest="human", default=None, type=int)
    parser.add_argument("--one", dest="one", action="store_true")
    parser.add_argument('--gammas', nargs='+', dest="rg", type=int, default=[10, 20, 40, 60, 80])

    # parser.add_argument("--substitution", dest="substitution", default="T", type=str)

    args = parser.parse_args()

    argparse_dict = vars(args)
    repo = Repo("./")
    argparse_dict["commit"] = str(repo.head.commit)

    os.makedirs(args.root, exist_ok=True)
    rac = os.path.split(args.name)[0]
    if rac == '':
        rac = "./"

    os.makedirs(rac, exist_ok=True)

    if not args.debug:
        f = open(os.devnull, 'w')
        sys.stdout = f

    with open(rac + '/params.json', 'w') as fp:
        json.dump(argparse_dict, fp, indent=True)

    if args.n_cpu is not None:
        n_cpu = args.n_cpu
    else:
        n_cpu = os.cpu_count()

    root = "data/raw/20170908-R9.5/"
    base_call = True
    rf = None
    human = False
    if args.target == "T":
        samf = "BTF_AG_ONT_1_FAH14273_A-select.sam"
        rf = "AG-basecalled/"

    if args.target == "TR":
        samf = ""
        rf = "../../../../../../../data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/AG_0-10"
        base_call = False

    if args.target == "B":
        samf = "BTF_AH_ONT_1_FAH14319_A-select.sam"
        rf = "AH-basecalled/"

    if args.target == "D":
        samf = ""
        rf = "AD-basecalled"
        base_call = False

    if args.target == "H_B":
        samf = ""
        rf = "Human-HR2/72"
        base_call = False
        human = True

    if args.target == "H_T":
        samf = ""
        rf = "Human_HQ/"
        base_call = False

    if rf is None:
        samf = ""
        rf = args.target
        base_call = False

    D = Dataset(samfile=root + samf,
                root_files=root + rf)
    D.metadata = argparse_dict
    D.substitution = args.target

    maxlen = args.maxlen

    if args.human is not None:
        if args.human == 1:
            human = True
            print("Human")
    ran = range(1, 11)
    if args.test_set:
        ran = range(11, 17)
    if args.range != []:
        ran = range(1, 17)
    D.populate(maxf=args.maxf, filter_not_alligned=True,
               filter_ch=ran, basecall=False, minion=False, arange=args.range,
               base_call=base_call)
    print("Popul", len(D.strands))
    D.strands = D.strands[:args.maxf]
    print("Popul", len(D.strands))

    from ..models.model import build_models

    def load_model():
        if args.Nbases == 4:
            mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}  # Modif
        elif args.Nbases == 5:
            mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B": 4, "N": 5}  # Modif
        elif args.Nbases == 8:
            mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B": 4,
                       "L": 5, "E": 6, "I": 7, "N": 8}  # Modif

        n_output_network = args.n_output_network

        subseq_size = args.ctc_length

        ctc_length = subseq_size
        input_length = ctc_length
        if n_output_network == 2:
            ctc_length = 2 * subseq_size

        n_feat = 4
        # if args.clean:
    #        n_feat = 3

        if args.sclean:
            n_feat = 1
        if args.norm2:
            n_feat = 2

        if args.allinfos:
            n_feat = args.maxleninf

        predictor, _ = build_models(args.size, nbase=args.Nbases - 4,
                                    ctc_length=ctc_length,
                                    input_length=None, n_output=n_output_network,
                                    lr=1, res=args.res, attention=args.attention,
                                    n_feat=n_feat, simple=args.simple, extra_output=args.extra_output, batchnorm=args.batchnorm,
                                    recurrent_dropout=args.dropout, one=args.one)

        if args.weights is not None:

            predictor.load_weights(args.weights)

        return predictor

    predictor = load_model()
    # except:
    # load from basecall

    def load_from_bc(strand, gamma):

        trans = strand.get_seq(f="no_basecall", window_size=args.window_size,
                               method=args.method, allinfos=args.allinfos,
                               maxlen=args.maxleninf, minlen=1, gamma=gamma, flatten=args.flatten)

        return [trans[:int(4 * maxlen)], None]

    if args.allinfos:
        if args.normed:
            fnorm = lambda x: scale_named4(x, maxleninf=args.maxleninf)
        else:
            fnorm = lambda x: scale_named4s(x, maxleninf=args.maxleninf)

    else:
        fnorm = scale_named2

    def compute_attributes(strand):
        # try:
        transfered = strand.transfered

        # strand.transfered_bc = copy.deepcopy(transfered)
        if len("".join(transfered["seq"]).replace("N", "")) > maxlen:
            transfered = transfered[:maxlen]

        # get the ref from transefered:
        from_ntwk = "".join(transfered["seq"]).replace("N", "")
        sub = "B"
        prop = from_ntwk.count("T") / (from_ntwk.count("T") + from_ntwk.count(sub) + 1e-7)
        ref = strand.get_ref(from_ntwk.replace(sub, "T"), correct=args.correct, human=human)
        # print(ref)
        if ref == "":
            print("Not alligned")
            return [None, len(from_ntwk) / len(transfered)]
        # allign the ref on the transefered
        bc_strand = from_ntwk.replace(sub, "T")
        al = strand.score(bc_strand, ref, all_info=True)
        # strand.score_bc_ref = al[2] / len(bc_strand)

        mapped_ref, correction = strand.give_map(
            "".join(transfered["seq"]).replace(sub, "T"), al[:2])

        def order(s1, s2):

            if prop < 0.5:
                s1 = s1.replace("T", sub)
                s2 = s2.replace("T", sub)
            else:
                s1 = s1.replace(sub, "T")
                s2 = s2.replace(sub, "T")
            if s1 != "N":
                return s1 + s2
            return s2 + s1

        new_ref = np.array([order(s, s1)
                            for s, s1 in zip(mapped_ref[::2], mapped_ref[1::2])])
        transfered["seq_ref"] = new_ref

        transfered["seq_ref_correction"] = np.array([order(s, s1)
                                                     for s, s1 in zip(correction[::2], correction[1::2])])
        strand.changed = True
        score = strand.score("".join(transfered["seq_ref"]).replace(
            "N", "").replace(sub, "T"), ref, all_info=False)
        print("ALL", al[2] / len(bc_strand))
        # print("".join(transfered["seq_ref"]).replace(
        #    "N", "").replace(sub, "T"))
        return transfered, al[2] / len(bc_strand), score, len(ref)

    # print(res)
    Density_network = {}
    B = {}
    Nb = {}
    All = {}
    Length = {}
    rg = args.rg
    for gamma in rg:
        Density_network[gamma] = []
        B[gamma] = []
        Nb[gamma] = []
        All[gamma] = []
        Length[gamma] = []

        for istrand, s in enumerate(D.strands):
            print(istrand)
            t = time.time()
            v = load_from_bc(s, gamma)
            print("TV", time.time() - t, len(v[0]))
            t = time.time()
            s.transfered = v[0]
            outputs = s.analyse_segmentation(predictor, fnorm(
                s.transfered), no2=args.n_output_network == 2)
            output = outputs[::, 0]
            if len(outputs) > 2:
                output2 = outputs[::, 1]
                s.transfered["seq"] = [s + s2 for s, s2 in zip(output, output2)]
            else:
                s.transfered["seq"] = [s + "N" for s in output]

            print("ADD to segments")
            s.segments["seq"] = s.transfered["seq"]

            print("predict", time.time() - t)

            Density_network[gamma].append(
                len("".join(s.transfered["seq"]).replace("N", "")) / len(s.transfered))
            seq = "".join(s.transfered["seq"])
            lseq = np.array([l for l in seq])
            # print(lseq)
            p = np.sum(lseq == "B") / (1 + np.sum(lseq == "T") + np.sum(lseq == "B"))
            B[gamma].append(p)
            Nb[gamma].append(len("".join(s.transfered["seq"]).replace("N", "")))
            Length[gamma].append(np.mean(s.transfered["length"] * s.sl))
            # Allignement
            v = compute_attributes(s)
            if v[0] is None:
                All[gamma].append(0)
            else:
                All[gamma].append(v[1])

        Density_network[gamma] = np.array(Density_network[gamma])
        B[gamma] = np.array(B[gamma])
        Nb[gamma] = np.array(Nb[gamma])
        All[gamma] = np.array(All[gamma])
        Length[gamma] = np.array(Length[gamma])

    np.set_printoptions(precision=2, suppress=True)

    print("Segment length")
    for gamma in rg:
        print(gamma, np.array(Length[gamma]))

    print("Base Density_network")
    for gamma in rg:
        print(gamma, Density_network[gamma])
    print("Percent B")
    for gamma in rg:
        print(gamma, B[gamma])
    print("Number of bases")
    for gamma in rg:
        print(gamma, Nb[gamma])
    print("Allignement")
    for gamma in rg:
        print(gamma, All[gamma])

    print("Sampling rate", s.sl)
    print("Summary")
    print("Gamma,Density,Percent,Nb,Score al", "Lengt   h")

    def mean(v):
        return float("%.2f" % np.mean(v))
    for gamma in rg:
        print(gamma, mean(Density_network[gamma]), mean(
            B[gamma]), mean(Nb[gamma]), mean(All[gamma]), mean(Length[gamma]))

    import pandas as pd
    df = pd.DataFrame({"gamma": rg,
                       "Density_base_mean": [np.mean(Density_network[gamma]) for gamma in rg],
                       "Percent_mean": [np.mean(B[gamma]) for gamma in rg],
                       "Nb_mean": [np.mean(Nb[gamma]) for gamma in rg],
                       "length_mean": [np.mean(Length[gamma]) for gamma in rg],
                       "Al_mean": [np.mean(All[gamma]) for gamma in rg],
                       "Density_base": [Density_network[gamma] for gamma in rg],
                       "Percent": [B[gamma] for gamma in rg],
                       "Nb": [Nb[gamma] for gamma in rg],
                       "Al": [All[gamma] for gamma in rg],
                       "length": [Length[gamma] for gamma in rg],
                       "Samplingrate": [s.sl for gamma in rg]})
    df.to_csv(args.name)

    """
        Density_alligned = []
        Length = []

        t = time.time()

        print("Cattr", time.time() - t)
        if v[0] is not None:
            s.transfered = v[0]
            s.bc_score = v[1]
            s.confirm_score = v[2]
            s.transfered["all"] = [np.array(a, dtype=np.float16) for a in s.transfered["all"]]
            s.transfered["mean"] = np.array(s.transfered["mean"], dtype=np.float16)
            s.transfered["stdv"] = np.array(s.transfered["stdv"], dtype=np.float16)
            s.segments = None
            Density_network.append(
                len("".join(s.transfered["seq"]).replace("N", "")) / len(s.transfered))
            Density_alligned.append(
                len("".join(s.transfered["seq_ref"]).replace("N", "")) / len(s.transfered))
            Length.append(np.mean([len(t) for t in s.transfered["all"]]))

        else:
            s.transfered = None
            Density_network.append(v[1])
            Length.append(np.mean([len(t) for t in s.segments["all"]]))
    """
    # print(output.shape, len(s.transfered))
    """
    data_x = []

    # except:

    #    return [None, None]

    # strand.transfered_seq = transfered

    # except:
    #     return [None, None]

    import _pickle as cPickle
    print("Writing on ", args.name)

    with open(args.name, "wb") as fich:
        cPickle.dump(D, fich)
    print("End writing")

    if args.info:
        print("########################")
        print("Infos:")
        print("Sampling rate", s.sl)
        print("Density of segments (network)", np.mean(Density_network))
        print(Density_network)
        print("Density of segments (alligned)", np.mean(Density_alligned))
        print(Density_alligned)
        print("lengths of segment", np.mean(Length))
        print(Length)
    """

    if K.backend() == 'tensorflow':

        K.clear_session()
