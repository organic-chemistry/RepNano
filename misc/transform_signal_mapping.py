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
import re
from taiyaki import alphabet, mapped_signal_files
MAPPED_SIGNAL_WRITER = mapped_signal_files.HDF5Writer
MAPPED_SIGNAL_READER = mapped_signal_files.HDF5Reader

from repnano.data.create_transition_matrix import list_transition, get_indexes

def clean_reads(reads, th=0.6, val="I",cano="T",lower_threshold=False,
                higher_threshold=False,motif=None,theoretical=None,calibration=None):
    print("Threshold",th)
    print("Loweth",lower_threshold)
    print("Higherth",higher_threshold)
    print("Cano",cano)
    p = []
    get_rid_of_read = []
    for k in reads.keys():
        seq=None
        # print(reads[k][1])
        percent = reads[k][1]
        #add = (np.array(list(reads[k][0]["seq"])) == cano) & np.isnan(percent)
        # print(np.sum(add),np.isnan(percent)[:10],(np.array(list(reads[k][0]["seq"]))=="T")[:10])
        #percent[add] = 0
        # print(reads[k][0]["seq"]=="T")
        if reads[k][0]["mapped_strand"] == "-":
            percent = percent[::-1]


        reads[k][0]["original_seq"] = ""+reads[k][0]["seq"]
        if (lower_threshold is not False) and (np.nanmean(percent)< lower_threshold):
            #Keep T only
            p.append(0)
            continue
        if (higher_threshold is not False) and (np.nanmean(percent)> higher_threshold):
            get_rid_of_read.append(k)
            continue



        #Assign with thereshold
        if type(th) is float:
            #print(np.nanmean(p[-1]))
            #print(reads[k][0]["seq"])
            #print(np.nanmean(percent))
            if np.nanmean(np.nanmean(percent)) >= th:
                #print("Modified",np.nanmean(percent))

                if motif == None:
                    seq = np.array(list(reads[k][0]["seq"]))
                    seq[seq == cano] = val
                else:
                    seq = reads[k][0]["seq"]
                    #print(str(seq))
                    indexes=np.array([_.start() for _ in re.finditer(motif,str(seq) )] ,dtype=int)
                    #print(indexes)
                    seq = np.array(list(reads[k][0]["seq"]))
                    seq[indexes]=val
                    #print(val,seq)
                    #print(indexes)
                reads[k][0]["seq"] = "".join(list(seq))
            else:
                pass
                #print("Unmodified")
            #exit()
        else:
            seq = np.array(list(reads[k][0]["seq"]))

            if calibration is not None:
                length=5
                x={"bases":np.array(list(reads[k][0]["seq"]))}
                indexes=get_indexes(x,length=length)
                thresholds=np.array(calibration.iloc[indexes]["percentile"])
                #print(thresholds)
                for i,(s,pt,t) in enumerate(zip(seq[length//2:],percent[length//2:],thresholds),length//2):
                    if seq[i]==cano:
                        if t not in [0,1]:
                            if pt>t:
                                seq[i] = val
                        else:
                            if np.random.rand()<theoretical:
                                seq[i]=val

                for i in list(range(length//2))+list(range(-length//2,0)):
                    if (seq[i]==cano) and (percent[i]>0.5):
                        seq[i]=val
            else:
                seq[(seq == cano) & (percent>0.5) ] = val

            reads[k][0]["seq"] = "".join(list(seq))



        if seq is None:
            p.append(np.nanmean(percent))
        else:
            #print(cano,val)
            #print(seq[:30])
            ncano = np.sum(np.array(seq) == cano)
            nval =  np.sum(np.array(seq) == val)
            #print(ncano,nval)
            p.append(nval/(nval+ncano+1e-7))


    #exit()
    for k in get_rid_of_read:
        reads.pop(k)
    return reads, p


def get_type(h5):
    if "Reads" in h5.keys():
        return "mega"
    return "rep"

def update_h5(file_n, bam_info, list_reads=[], maxi=2,
              new_alphabet=None, mod_long_names=None,
              val_new="B",cano="T",
              hout=None,global_alphabet=None,
              collapsed=None,
              filter_section=False,subsample=[[0,0.1],[1,0.25]]):
    with MAPPED_SIGNAL_READER(file_n) as f:

        # print(f.attrs["collapse_alphabet"])
        # print(f.attrs["mod_long_names"])
        # print(f.attrs["version"])

        #if new_alphabet is None:
        #    new_alphabet = f.get_alphabet_information().alphabet
        collapsed = f.get_alphabet_information().collapse_alphabet
        print("Old alphabet",f.get_alphabet_information().alphabet)

        #conv = np.zeros(len(global_alphabet.alphabet),dtype=np.int16)
        #for i,l in enumerate(new_alphabet):
        #    conv[i] = global_alphabet.alphabet.index(l)

        def get_seq(array,collapsed_alphabet):
            new=np.zeros(len(array),dtype=np.str)
            for i,v in enumerate(collapsed_alphabet):
                new[array==i] = v
            return "".join(list(new))

        #print(new_alphabet)
        #print(collapsed)

        Exp = {}
        ik = 0
        p=[]
        error = 0
        skiped = 0
        for k in f.get_read_ids():
            if filter_section:
                if k in bam_info.keys() and bam_info[k][2][0] is None:
                    continue
            read = f.get_read(k)


            if k in bam_info.keys():
                seq = np.array(list(bam_info[k][0]["seq"]))
                assert (len(seq) == len(read.Reference))
                assert bam_info[k][0]["original_seq"][:100] == get_seq(read.Reference[:100],collapsed)
                r = np.zeros(len(seq), dtype=np.int16)
                #print(k,seq)
                for l, val in zip(list(new_alphabet), range(0, len(new_alphabet))):
                    r[seq == l] = val
                    read.Reference = r
            else:
                continue
                #print("bo",read.Reference)
                #read.Reference = conv[read.Reference]
                #print("af",read.Reference)
                #print("Found")
            """
    if not len(exp["Dacs"])>=max(exp["Ref_to_signal"]):
                print(len(exp["Dacs"]),max(exp["Ref_to_signal"]))
                print(min(exp["Ref_to_signal"]))
            """
            if filter_section:
                if  k in bam_info.keys() and bam_info[k][2][0] is not None:
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

            list_seq = np.array(read.Reference)
            i_val = global_alphabet.alphabet.index(val_new)
            cano_val = global_alphabet.alphabet.index(cano)
            #print(read.Reference)
            new = np.sum(list_seq==i_val)
            old = np.sum(list_seq==cano_val)
            #print(new,old)
            percent_actual_value = new/(new+old)
            skip=False
            if subsample != []:
                for subv in subsample:
                    if subv[0] == percent_actual_value:
                        if np.random.rand()>subv[1]:
                            #print("Sub")
                            skip=True
                            skiped += 1
            if skip:
                continue
            try:
                hout.write_read(read.get_read_dictionary())
            except ValueError:
                error += 1

            #if percent_actual_value != 0:
            #    print(k)
            #    print(new_alphabet)
            #    print(seq)
            #    print(percent_actual_value,list_seq)
            p.append(percent_actual_value)

            # b = exp["Dacs"][exp["Ref_to_signal"]-1]
            if maxi is not None and ik + skiped >= maxi-1:
                break
            ik += 1
    print("N redundant",error)
    print("Skiped for subpsample",skiped)
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
    parser.add_argument('--max',type=int,default=None)
    parser.add_argument('--calibration', action="store_true" )



    args = parser.parse_args()
    """
    with open(args.training_file,"r") as f:
        training_file = json.loads("".join(f.readlines()))
    """

    dataset = pd.read_csv(args.dataset,sep=";")
    #print("data",dataset)
    #print(dataset.columns)
    print(args.alphabet,args.collapsed_alphabet)
    alpha = alphabet.AlphabetInfo(args.alphabet, args.collapsed_alphabet,args.mod_long_names)

    with  MAPPED_SIGNAL_WRITER(args.output, alpha) as hout:

        for index, row in dataset.iterrows():

            print(row)

            new_alphabet = row["new_alphabet"]
            mod_long_names = row["mod_long_names"]
            canonical = row.get("canonical","T")

            threshold = row["threshold"]
            filter_section = row["filter_section"]
            lower_threshold = row.get("lower_threshold",False) #Under set to 0
            higher_threshold = row.get("higher_threshold",False)#higher set to higvalue
            subsample = row.get("subsample","[]")
            subsample = ast.literal_eval(subsample)
            motif=row.get("motif",None)
            theoretical = row.get("theoretical",None)
            if motif=="" or type(motif)!=str:
                motif=None

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

            if type(lower_threshold) == str:
                if lower_threshold == "False":
                    lower_threshold = False
                else:
                    try:
                        lower_threshold = float(lower_threshold)
                    except:
                        lower_threshold = bool(lower_threshold)

            if type(higher_threshold) == str:
                if higher_threshold == "False":
                    higher_threshold = False
                else:
                    try:
                        higher_threshold = float(higher_threshold)
                    except:
                        higher_threshold = bool(higher_threshold)

            modified_base = row["modified_base"]
            key = row["key"]


            if type(new_alphabet) == float:
                new_alphabet = None
            if type(mod_long_names) == float:
                mod_long_names = None


            print("threshold",threshold)
            print("filter_section",filter_section)
            print("subsample",subsample)
            print(new_alphabet,mod_long_names)



            if threshold is not None:
                root_bam = args.root_h5 + f"/{key}/mod_mappings.bam"

                reads = load_read_bam(root_bam,
                                      filter_b=0, n_b=1,fill_nan=True,maxi=args.max)
                print(len(reads))

                pylab.clf()
                pylab.hist([np.nanmean(pi[1]) for pi in reads.values() ],bins=100,range=[0,1])
                pylab.savefig(args.root_h5+f"/{key}/original_histo_{modified_base}.png")

                calibration_file=root_bam.replace(".bam","_calibration.csv")
                calibration=None
                if os.path.exists(calibration_file):
                    calibration = pd.read_csv(calibration_file)
                    if theoretical == None:
                        calibration = None
                    print(calibration)
                new_reads, p = clean_reads(reads, th=threshold, val=modified_base,
                                           cano=canonical,
                                           lower_threshold=lower_threshold,
                                           higher_threshold=higher_threshold,
                                           motif=motif,calibration=calibration,
                                           theoretical=theoretical)
                root_map = args.root_h5 + f"/{key}/signal_mappings.hdf5"
                p = update_h5(file_n=root_map,
                                bam_info=new_reads, maxi=args.max,
                              new_alphabet=args.alphabet,
                              mod_long_names=mod_long_names,
                                val_new=modified_base,cano=canonical,hout=hout,global_alphabet=alpha,
                              filter_section=filter_section,subsample=subsample)



                #with  MAPPED_SIGNAL_WRITER(args.output, merge_alphabet_info) as hout:
                #    for infile in args.input:
                #        with MAPPED_SIGNAL_READER(infile) as hin:

                pylab.clf()
                pylab.hist(p,bins=100,range=[0,1])
                pylab.savefig(args.root_h5+f"/{key}/histo_{modified_base}.png")
