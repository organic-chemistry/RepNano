if __name__ == "__main__":
    import argparse
    import json
    from git import Repo
    import os
    from multiprocessing import Pool
    import numpy as np

    from ..data.dataset import Dataset, NotAllign
    from ..features.helpers import scale_simple, scale_named
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', dest="window_size", type=int, choices=[4, 5, 8], default=5)
    parser.add_argument('--root', type=str, default="data/training/")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--allinfos', dest='allinfos', action='store_true')

    parser.add_argument('--size', type=int, default=20)
    parser.add_argument('--maxlen', type=int, default=36)
    parser.add_argument('--minlen', type=int, default=1)

    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=None)
    parser.add_argument("--name", dest="name", type=str, default='dataset.pick')
    parser.add_argument("--target", dest="target", type=str, default='T')
    parser.add_argument("--test-set", dest="test_set", action="store_true")
    parser.add_argument("--range", dest="range", nargs='+', default=[], type=float)
    parser.add_argument("--method", dest="method",
                        choices=["FW", "TV", "TV45", "TV25", "TV5", "TVb"])
    parser.add_argument('--correct', dest='correct', action='store_true')
    parser.add_argument('--no-align', dest='realign', action='store_false')
    parser.add_argument('--human', dest='human', action='store_true')

    parser.add_argument('--gamma', type=float, default=40)

    #parser.add_argument("--substitution", dest="substitution", default="T", type=str)

    args = parser.parse_args()

    argparse_dict = vars(args)
    repo = Repo("./")
    argparse_dict["commit"] = str(repo.head.commit)

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(os.path.split(args.name)[0], exist_ok=True)

    with open(os.path.split(args.name)[0] + '/params.json', 'w') as fp:
        json.dump(argparse_dict, fp, indent=True)

    if args.n_cpu is not None:
        n_cpu = args.n_cpu
    else:
        n_cpu = os.cpu_count()

    root = "data/raw/20170908-R9.5/"
    base_call = True
    if args.target == "T":
        samf = "BTF_AG_ONT_1_FAH14273_A-select.sam"
        rf = "AG-basecalled/"
    if args.target == "B":
        samf = "BTF_AH_ONT_1_FAH14319_A-select.sam"
        rf = "AH-basecalled/"

    if args.target == "D":
        samf = ""
        rf = "AD-basecalled"
        base_call = False

    if args.target == "H_B":
        samf = ""
        rf = "Human_AR/"
        base_call = False

    if args.target == "H_T":
        samf = ""
        rf = "Human_HQ/"
        base_call = False

    if args.target == "H_B_B":
        root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/"
        samf = ""
        rf = "Albacore-human/HR/workspace/"

    if args.target == "H_T_B":
        root = "/data/bioinfo@borvo/users/jarbona/deepnano5bases/data/raw/"
        samf = ""
        rf = "Albacore-human/HQ/workspace/"

    D = Dataset(samfile=root + samf,
                root_files=root + rf)
    D.metadata = argparse_dict
    D.substitution = args.target

    maxf = None
    maxlen = 20000
    if args.test:
        maxf = 2
        maxlen = 1000

    ran = range(1, 11)
    if args.test_set:
        ran = range(11, 17)
    if args.range != []:
        ran = range(1, 17)
    D.populate(maxf=maxf, filter_not_alligned=True,
               filter_ch=ran, basecall=False, minion=False, arange=args.range,
               base_call=base_call, samf=samf)
    # load from basecall

    def load_from_bc(strand):
        if base_call:
            try:
                bc = strand.get_seq(f="BaseCall")
            except (NotAllign, KeyError) as e:
                bc = [None]

            if samf != "":
                minion = strand.get_seq(f="Minion", correct=True)
                return [bc, minion]
            else:
                return [bc, None]
        else:
            trans = strand.get_seq(
                f="no_basecall", window_size=args.window_size,
                method=args.method, allinfos=True, maxlen=args.maxlen, minlen=args.minlen)

            return [trans, None]

    print(len(D.strands))
    with Pool(n_cpu) as p:
        res = p.map(load_from_bc, D.strands)

    pop = []
    for istrand, (v, s) in enumerate(zip(res, D.strands)):
        if base_call:
            if v[0][0] is not None:
                s.signal_bc, s.seq_from_basecall, s.imin, s.raw, s.to_match, s.sampling_rate = v[0]
                s.seq_from_minion = v[1]
                s.to_match = ""
            else:
                pop.append(istrand)
        else:
            s.transfered = v[0]

    for istrand in pop[::-1]:
        D.strands.pop(istrand)

    data_x = []

    def compute_attributes(strand):
        # try:
        strand.segmentation(w=args.window_size, method=args.method,
                            allinfos=args.allinfos, maxlen=args.maxlen, minlen=args.minlen,
                            gamma=args.gamma, totallen=None)

        # print(strand.segments.columns)
        transfered = strand.transfer(strand.signal_bc, strand.segments, allinfos=args.allinfos)
        # print(transfered.columns)
        # strand.transfered_bc = copy.deepcopy(transfered)
        init_len = len(transfered)
        if len("".join(transfered["seq"]).replace("N", "")) > maxlen:
            transfered = transfered[:maxlen]
        # get the ref from transefered:

        if args.realign:
            ref = strand.get_ref("".join(transfered["seq"]).replace(
                "N", ""), correct=args.correct, human=args.human)

            if ref == "":
                return [None, None]
            # allign the ref on the transefered
            bc_strand = "".join(transfered["seq"]).replace("N", "")
            al = strand.score(bc_strand, ref, all_info=True)
            # strand.score_bc_ref = al[2] / len(bc_strand)

            mapped_ref, correction = strand.give_map("".join(transfered["seq"]), al[:2])

            def order(s1, s2):
                if s1 != "N":
                    return s1 + s2
                return s2 + s1
            transfered["seq_ref"] = np.array([order(s, s1)
                                              for s, s1 in zip(mapped_ref[::2], mapped_ref[1::2])])
            transfered["seq_ref_correction"] = np.array([order(s, s1)
                                                         for s, s1 in zip(correction[::2], correction[1::2])])
            strand.changed = True
            bc_score = al[2] / len(bc_strand)
            confirm_score = strand.score("".join(transfered["seq_ref"]).replace(
                "N", ""), ref, all_info=False)

        else:
            transfered["seq_ref"] = transfered["seq"]
            transfered["seq_ref_correction"] = ["NN" for _ in transfered["seq"]]

            print(init_len, len(transfered))
            bc_score = 0
            confirm_score = 0

        # strand.transfered_seq = transfered

        return transfered, bc_score, confirm_score, ""
        # except:
        #    return [None, None]
    alligned = 0
    if base_call:
        with Pool(n_cpu) as p:
            res = p.map(compute_attributes, D.strands)
        for v, s in zip(res, D.strands):
            if v[0] is not None:
                s.transfered = v[0]
                s.bc_score = v[1]
                s.confirm_score = v[2]
                if hasattr(s, "segments"):
                    s.segments = None
                del(s.signal_bc)
                alligned += 1
                print(v[1], v[2])
            else:
                s.transfered = None
    print("Nmol alligned", alligned)
    import _pickle as cPickle
    with open(args.name, "wb") as fich:
        cPickle.dump(D, fich)
    """
    for strand in D.strands:

        if args.sclean:
            data_x.append(scale_simple(transfered))
        else:
            data_x.append(scale_named(transfered))

        data_y.append([mapping[b] for b in transfered["seq"]])
    """
