import tempfile
import ast
from repnano.features.bam_tools import load_read_bam
import argparse

import numpy as np
import h5py
def get_h5_p(file_n, maxi=None,cano="T",mods=["B","I"]):
    with h5py.File(file_n, "r") as f:

        print("After",dict(f.attrs.items()))

        new_alphabet = f.attrs["alphabet"]
        to_pop = []
        for m in mods:
            if m not in new_alphabet:
                print(f"Warning {m} not in alphabet")
                print(f"Plot for {m} will not be produceds")
                to_pop.append(m)
        mods.remove(m)
        Exp = {}
        ik = 0
        p={m:[] for m in mods}
        for k in f["Reads"]:
            read = f["Reads"][k]

            exp = dict(read.attrs.items())
            # print(exp)
            # print(read.keys())
            exp["Reference"] = np.array(read["Reference"])
            exp["Ref_to_signal"] = np.array(read["Ref_to_signal"])
            exp["Dacs"] = np.array(read["Dacs"])
            if "path" in read:
                exp["path"] = np.array(read["path"])
            Exp[exp["read_id"]] = exp
            # print(exp["read_id"])

            list_seq=exp["Reference"]
            #print(list_seq)
            for mod in mods:
                i_val = new_alphabet.index(mod)
                cano_val = new_alphabet.index(cano)
                p[mod].append(np.sum(list_seq==i_val)/(np.sum(list_seq==i_val)+np.sum(list_seq==cano_val)+1))

            # b = exp["Dacs"][exp["Ref_to_signal"]-1]
            if maxi is not None and ik > maxi:
                break
            ik += 1
    return p,mods

if __name__ =="__main__":
    import pylab
    import matplotlib as mpl

    mpl.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str )
    parser.add_argument('--mods', nargs='+', type=str,default=["B","I"])

    args = parser.parse_args()

    p,mods = get_h5_p(args.h5,mods=args.mods)
    #print(p)
    for mod in mods:
        pylab.clf()
        pylab.hist(p[mod], bins=100,label=f"Total reads {len(p[mod])}")
        pylab.legend()
        pylab.savefig(f"{mod}_histo.png")


