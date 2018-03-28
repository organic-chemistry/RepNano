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
    parser.add_argument('--maxlen', type=int, default=35)
    parser.add_argument('--minlen', type=int, default=1)

    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=None)
    parser.add_argument("--name", dest="name", type=str, default='dataset.pick')
    parser.add_argument("--target", dest="target", type=str, default='T')
    parser.add_argument("--test-set", dest="test_set", action="store_true")
    parser.add_argument("--range", dest="range", nargs='+', default=[], type=float)
    parser.add_argument("--method", dest="method",
                        choices=["FW", "TV", "TV45", "TV25", "TV5", "TVb"])

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
               base_call=base_call)
    # load from basecall

    def load_from_bc(strand):
        if samf != "":
            try:
                bc = strand.get_seq(f="BaseCall")
            except NotAllign:
                bc = [None]

            minion = strand.get_seq(f="Minion", correct=True)
            return [bc, minion]
        else:
            trans = strand.get_seq(
                f="no_basecall", window_size=args.window_size, method=args.method)

            return [trans, None]

    print(len(D.strands))
    with Pool(n_cpu) as p:
        res = p.map(load_from_bc, D.strands)

    pop = []
    for istrand, (v, s) in enumerate(zip(res, D.strands)):
        if samf != "":
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
        try:
            strand.segmentation(w=args.window_size, method=args.method,
                                allinfos=args.allinfos, maxlen=args.maxlen, minlen=args.minlen)

            transfered = strand.transfer(strand.signal_bc, strand.segments)
            # strand.transfered_bc = copy.deepcopy(transfered)
            if len("".join(transfered["seq"]).replace("N", "")) > maxlen:
                transfered = transfered[:maxlen]
            # get the ref from transefered:
            ref = strand.get_ref("".join(transfered["seq"]).replace("N", ""), correct=False)
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

            # strand.transfered_seq = transfered

            return transfered, al[2] / len(bc_strand), strand.score("".join(transfered["seq_ref"]).replace(
                "N", ""), ref, all_info=False), len(ref)
        except:
            return [None, None]

    if samf != "":
        with Pool(n_cpu) as p:
            res = p.map(compute_attributes, D.strands)
        for v, s in zip(res, D.strands):
            if v[0] is not None:
                s.transfered = v[0]
                s.bc_score = v[1]
                s.confirm_score = v[2]
                del(s.signal_bc)
            else:
                s.transfered = None
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
