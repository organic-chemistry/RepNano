from repnano.data.create_transition_matrix import list_transition, get_indexes
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


def clean_reads(reads, length=5,th=0.6, val="I",cano="T",lower_threshold=False,higher_threshold=False,motif=None,name=""):
    print("Threshold",th)
    print("Loweth",lower_threshold)
    print("Higherth",higher_threshold)
    print("Cano",cano)
    p = []
    get_rid_of_read = []

    list_trans,ltv = list_transition(length)
    dic = [ [] for i in range(len(list_trans))]
    print(len(dic),length)
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
            continue
        if (higher_threshold is not False) and (np.nanmean(percent)> higher_threshold):
            continue

        x={"bases":np.array(list(reads[k][0]["seq"]))}

        indexes=get_indexes(x,length=length)
        #Assign with threshold
        """
        seq = reads[k][0]["seq"]
        #print(str(seq))
        indexes=np.array([_.start() for _ in re.finditer(motif,str(seq) )] ,dtype=int)
        #print(indexes)
        seq = np.array(list(reads[k][0]["seq"]))
        seq[indexes]=val
        reads[k][0]["seq"] = "".join(list(seq))
        """
        for inde,perc in zip(indexes,percent[length//2:]):
            dic[inde].append(perc)

    #exit()
    print("Sup percentile",100-th*100)
    percentiles=[]
    Ne=[]
    for ind,penta in enumerate(list_trans) :
        percentiles.append(np.percentile(dic[ind],100-th*100))
        Ne.append(len(dic[ind]))
    pd.DataFrame({"seq":["".join(l) for l in list_trans],"percentile":percentiles,"nsample":Ne}).to_csv(name,index=False)
    return reads, p


if __name__ == "__main__":
        import pylab
        import matplotlib as mpl

        mpl.use("Agg")
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str )
        parser.add_argument('--root_h5', type=str )
        parser.add_argument('--max',type=int,default=None)



        args = parser.parse_args()
        """
        with open(args.training_file,"r") as f:
            training_file = json.loads("".join(f.readlines()))
        """

        dataset = pd.read_csv(args.dataset,sep=";")
        #print("data",dataset)
        #print(dataset.columns)



        for index, row in dataset.iterrows():

            print(row)

            new_alphabet = row["new_alphabet"]
            mod_long_names = row["mod_long_names"]
            canonical = row.get("canonical","T")

            threshold = row["threshold"]
            theoretical = row["theoretical"]
            filter_section = row["filter_section"]
            lower_threshold = row.get("lower_threshold",False) #Under set to 0
            higher_threshold = row.get("higher_threshold",False)#higher set to higvalue
            subsample = row.get("subsample","[]")
            subsample = ast.literal_eval(subsample)
            motif=row.get("motif",None)
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
            root_bam = args.root_h5 + f"/{key}/mod_mappings.bam"

            reads = load_read_bam(root_bam,
                      filter_b=0, n_b=1,fill_nan=True,maxi=args.max,
                      calibration=False)

            clean_reads(reads,length=5, th=theoretical, val=modified_base,
                           cano=canonical,
                           lower_threshold=lower_threshold,
                           higher_threshold=higher_threshold,
                           motif=motif,name=root_bam.replace(".bam","_calibration.csv"))


    #load bwa and get probability of modified base per transition
    #compute threshold
