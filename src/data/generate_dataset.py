if __name__ == "__main__":
    import argparse
    import json
    from git import Repo
    import os
    from multiprocessing import Pool
    import numpy as np

    from ..data.dataset import Dataset, NotAllign
    from ..features.helpers import scale_simple, scale_named

    parser = argparse.ArgumentParser()
    parser.add_argument('--window-size', dest="window_size", type=int, choices=[4, 5, 8], default=5)
    parser.add_argument('--root', type=str, default="data/training/")
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--size', type=int, default=20)
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--n-cpu", dest="n_cpu", type=int, default=None)
    parser.add_argument("--name", dest="name", type=str, default='dataset.pick')
    parser.add_argument("--target", dest="target", type=str, default='T')

    args = parser.parse_args()

    argparse_dict = vars(args)
    repo = Repo("./")
    argparse_dict["commit"] = str(repo.head.commit)

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(os.path.split(args.name)[0], exist_ok=True)

    with open(args.root + '/params.json', 'w') as fp:
        json.dump(argparse_dict, fp, indent=True)

    if args.n_cpu is not None:
        n_cpu = args.n_cpu
    else:
        n_cpu = os.cpu_count()

    root = "data/raw/20170908-R9.5/"
    if args.target == "T":
        samf = "BTF_AG_ONT_1_FAH14273_A-select.sam"
        rf = "AG-basecalled/"
    if args.target == "B":
        samf = "BTF_AH_ONT_1_FAH14319_A-select.sam"
        rf = "AH-basecalled/"
    D = Dataset(samfile=root + samf,
                root_files=root + rf)

    maxf = None
    maxlen = 20000
    if args.test:
        maxf = 12
        maxlen = 1000

    D.populate(maxf=maxf, filter_not_alligned=True,
               filter_ch=range(1, 11), basecall=False, minion=False)

    # load from basecall
    def load_from_bc(strand):
        try:
            bc = strand.get_seq(f="BaseCall")
        except NotAllign:
            bc = [None]

        minion = strand.get_seq(f="Minion", correct=True)
        return [bc, minion]
    with Pool(n_cpu) as p:
        res = p.map(load_from_bc, D.strands)

    pop = []
    for istrand, (v, s) in enumerate(zip(res, D.strands)):
        if v[0][0] is not None:
            s.signal_bc, s.seq_from_basecall, s.imin, s.raw, s.to_match, s.sampling_rate = v[0]
            s.seq_from_minion = v[1]
            s.to_match = ""
        else:
            pop.append(istrand)

    for istrand in pop[::-1]:
        D.strands.pop(istrand)

    data_x = []

    def compute_attributes(strand):
        try:
            strand.segmentation(w=args.window_size)

            transfered = strand.transfer(strand.signal_bc, strand.segments)
            # strand.transfered_bc = copy.deepcopy(transfered)
            if len("".join(transfered["seq"]).replace("N", "")) > maxlen:
                transfered = transfered[:maxlen]
            # get the ref from transefered:
            ref = strand.get_ref("".join(transfered["seq"].replace("N", "")), correct=True)
            if ref == "":
                return [None, None]
            # allign the ref on the transefered
            bc_strand = "".join(transfered["seq"]).replace("N", "")
            al = strand.score(bc_strand, ref, all_info=True)
            strand.score_bc_ref = al[2] / len(bc_strand)

            mapped_ref, correction = strand.give_map("".join(transfered["seq"]), al[:2])

            transfered["seq_ref"] = np.array([s for s in mapped_ref])
            transfered["seq_ref_correction"] = np.array([s for s in correction])
            strand.changed = True

        # strand.transfered_seq = transfered

            return transfered, al[2] / len(bc_strand), strand.score("".join(transfered["seq"]).replace(
                "N", ""), ref, all_info=False), len(ref)
        except:
            return [None, None]

    with Pool(n_cpu) as p:
        res = p.map(compute_attributes, D.strands)
    for v, s in zip(res, D.strands):
        if v[0] is not None:
            s.transfered = v[0]
            s.bc_score = v[1]
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
