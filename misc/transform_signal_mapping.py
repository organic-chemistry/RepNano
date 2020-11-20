import pandas as pd
import glob
import os
import subprocess
import argparse
import os
import json
import tempfile
import ast
from repnano.features.bam_tools import load_read_bam
import numpy as np
import h5py
from taiyaki import alphabet, mapped_signal_files
MAPPED_SIGNAL_WRITER = mapped_signal_files.HDF5Writer
MAPPED_SIGNAL_READER = mapped_signal_files.HDF5Reader

def clean_reads(reads, th=0.6, val="I",cano="T"):
    print("Threshold",th)
    p = []
    for k in reads.keys():
        # print(reads[k][1])
        percent = reads[k][1]
        add = (np.array(list(reads[k][0]["seq"])) == cano) & np.isnan(percent)
        # print(np.sum(add),np.isnan(percent)[:10],(np.array(list(reads[k][0]["seq"]))=="T")[:10])
        percent[add] = 0
        # print(reads[k][0]["seq"]=="T")
        p.append(np.nanmean(percent))
        reads[k][0]["original_seq"] = ""+reads[k][0]["seq"]

        #Assign with thereshold
        if type(th) is float:
            #print(np.nanmean(p[-1]))
            #print(reads[k][0]["seq"])
            if np.nanmean(p[-1]) > th:
                #print("Modified")
                seq = np.array(list(reads[k][0]["seq"]))
                seq[seq == cano] = val
                reads[k][0]["seq"] = "".join(list(seq))
            else:
                pass
                #print("Unmodified")
            #exit()
        else:
            #Assign with guppy output
            seq = np.array(list(reads[k][0]["seq"]))
            seq[(seq == cano) & (percent>0.5) ] = val
            reads[k][0]["seq"] = "".join(list(seq))
    #exit()
    return reads, p


def get_type(h5):
    if "Reads" in h5.keys():
        return "mega"
    return "rep"

def update_h5(file_n, bam_info, list_reads=[], maxi=2,
              new_alphabet=None, mod_long_names=None,val_new="B",cano="T",hout=None,global_alphabet=None,filter_section=False):
    with MAPPED_SIGNAL_READER(file_n) as f:

        # print(f.attrs["collapse_alphabet"])
        # print(f.attrs["mod_long_names"])
        # print(f.attrs["version"])

        if new_alphabet is None:
            new_alphabet = f.get_alphabet_information().alphabet
        collapsed = f.get_alphabet_information().collapse_alphabet

        conv = np.zeros(len(global_alphabet.alphabet),dtype=np.int16)
        for i,l in enumerate(new_alphabet):
            conv[i] = global_alphabet.alphabet.index(l)

        def get_seq(array,collapsed_alphabet):
            new=np.zeros(len(array),dtype=np.str)
            for i,v in enumerate(collapsed_alphabet):
                new[array==i] = v
            return "".join(list(new))



        Exp = {}
        ik = 0
        p=[]
        for k in f.get_read_ids():
            read = f.get_read(k)

            if k in bam_info.keys():
                seq = np.array(list(bam_info[k][0]["seq"]))
                assert (len(seq) == len(read.Reference))
                assert bam_info[k][0]["original_seq"][:100] == get_seq(read.Reference[:100],collapsed)
                r = np.zeros(len(seq), dtype=np.int16)
                for l, val in zip(list(new_alphabet), range(0, len(new_alphabet))):
                    r[seq == l] = val
                    read.Reference = r
                read.Reference = conv[read.Reference]
                #print("Found")
            """            
    if not len(exp["Dacs"])>=max(exp["Ref_to_signal"]):
                print(len(exp["Dacs"]),max(exp["Ref_to_signal"]))
                print(min(exp["Ref_to_signal"]))
            """
            if filter_section:
                if bam_info[k][2][0] is not None:
                    start,end = bam_info[k][2]
                    #print(read.Reference)
                    read.Reference = read.Reference[start:end]
                    start_d = read.Ref_to_signal[start]
                    end_d = read.Ref_to_signal[end]
                    read.Ref_to_signal = read.Ref_to_signal[start:end]
                    read.Ref_to_signal-= start_d
                    read.Dacs = read.Dacs[start_d:end_d]

                else:
                    continue

            hout.write_read(read.get_read_dictionary())
            list_seq=read.Reference
            i_val = global_alphabet.alphabet.index(val_new)
            cano_val = global_alphabet.alphabet.index(cano)
            p.append(np.sum(list_seq==i_val)/(np.sum(list_seq==i_val)+np.sum(list_seq==cano_val)))

            # b = exp["Dacs"][exp["Ref_to_signal"]-1]
            if maxi is not None and ik > maxi:
                break
            ik += 1
    return p

if __name__ == "__main__":

    import pylab
    import matplotlib as mpl

    mpl.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str )
    parser.add_argument('--root_h5', type=str )
    parser.add_argument('--output', type=str )
    parser.add_argument('--alphabet', type=str )
    parser.add_argument('--collapsed_alphabet', type=str )
    parser.add_argument('--mod_long_names', nargs='+',type=str)

    args = parser.parse_args()
    """
    with open(args.training_file,"r") as f:
        training_file = json.loads("".join(f.readlines()))
    """

    dataset = pd.read_csv(args.dataset,sep=";")
    #print("data",dataset)
    #print(dataset.columns)

    alpha = alphabet.AlphabetInfo(args.alphabet, args.collapsed_alphabet,args.mod_long_names)

    with  MAPPED_SIGNAL_WRITER(args.output, alpha) as hout:

        for index, row in dataset.iterrows():

            print(row)

            new_alphabet = row["new_alphabet"]
            mod_long_names = row["mod_long_names"]
            threshold = row["threshold"]
            filter_section = row["filter_section"]
            if type(filter_section) == str:
                if filter_section == "False":
                    filter_section = False
                else:
                    filter_section = True
            #print(threshold,type(threshold))
            if type(threshold) == str:
                if threshold == "None":
                    threshold = None
                elif threshold == "False":
                    threshold = False
                else:
                    try:
                        threshold = float(threshold)
                    except:
                        threshold = bool(threshold)

            modified_base = row["modified_base"]
            key = row["key"]


            if type(new_alphabet) == float:
                new_alphabet = None
            if type(mod_long_names) == float:
                mod_long_names = None


            print("threshold",threshold)
            print("filter_section",filter_section)
            print(new_alphabet,mod_long_names)



            if threshold is not None:
                root_bam = args.root_h5 + f"/{key}/mod_mappings.bam"

                reads = load_read_bam(root_bam,
                                      filter_b=0, n_b=1)

                new_reads, p = clean_reads(reads, th=threshold, val=modified_base,cano="T")
                root_map = args.root_h5 + f"/{key}/signal_mappings.hdf5"
                p = update_h5(file_n=root_map,
                                bam_info=new_reads, maxi=None, new_alphabet=new_alphabet, mod_long_names=mod_long_names,
                                val_new=modified_base,cano="T",hout=hout,global_alphabet=alpha,filter_section=filter_section)



                #with  MAPPED_SIGNAL_WRITER(args.output, merge_alphabet_info) as hout:
                #    for infile in args.input:
                #        with MAPPED_SIGNAL_READER(infile) as hin:

                pylab.clf()
                pylab.hist(p,bins=100)
                pylab.savefig(args.root_h5+f"/{key}/histo_{modified_base}.png")


