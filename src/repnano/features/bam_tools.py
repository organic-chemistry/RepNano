import pysam
import tqdm
import numpy as np

def convert_to_coordinate(seq,Ml,Mm,which="T"):
    #Ml base proba
    #Mm number of base to skip
    Mmc=Mm.copy()
    result = np.zeros((len(seq)))+np.nan
    n_which = 0
    for bi,s in enumerate(seq):
        if s == which:
            #print(bi,len(seq))
            if n_which>len(Mmc)-1:
                break
            skip = Mmc[n_which]
            if skip == 0:
                result[bi] = Ml[n_which]/255
                n_which += 1
            else:
                Mmc[n_which]-=1
    return result

import pandas as pd
def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())

def get_longest_low(v_mono):
    #print("Here")
    monov = smooth(v_mono, 1000)
    sum_b = np.nansum((smooth(v_mono, 100)) > 0.5)
    if np.isnan(sum_b) or (not sum_b > 30):
        return None,None
    selected = (monov > 0.015) & (monov < 0.35)
    found=False
    while np.sum(selected) > 1000:
        st = "".join([str(int(s)) for s in selected])
        sp = st.split("0")
        longest = np.argmax([len(ss) for ss in sp])

        def getl(ss):
            if ss == "":
                return 1
            else:
                return len(ss)

        start = int(sum([getl(ss) for ss in sp[:longest]]))
        end = int(start + len(sp[longest]))
        #print(start,end)
        if np.any(monov[start:end] > 0.1):
            found=True
            break
        else:
            selected[start:end] = 0
    if found:
        return start, end
    else:
        return None, None

def load_read_bam(bam,filter_b=0.5,n_b=5000,verbose=False):
    samfile = pysam.AlignmentFile(bam, "r")#,check_sq=False)

    Read ={}

    for ir,read in tqdm.tqdm(enumerate(samfile)):
        if verbose:
            print(ir,read)
        seq,Ml,Mm = read.get_forward_sequence(),read.get_tag("Ml"),[int(v) for v in read.get_tag("Mm")[:-1].split(",")[1:]]
        attr={}
        if read.is_reverse:
            attr["mapped_strand"] = "-"
        else:
            attr["mapped_strand"] = "+"
        attr["mapped_chrom"] = "chr%i"%read.reference_id
        pos = read.get_reference_positions()
        attr["mapped_start"] = pos[0]
        attr["mapped_end"] = pos[-1]
        attr["seq"]=seq
        val=convert_to_coordinate(seq,Ml,Mm)
        val[np.isnan(val) & (np.array(list(seq)) == "T")] = 0
        Nn=val
        if np.sum(np.array(Nn)>filter_b) > n_b:
            Read[read.query_name] = [attr,Nn,get_longest_low(Nn)]
        #break
    return Read

#load_read_bam("../../../debug/debug_GPU_ref/bam_runid_1ca9cbe0f2c9d4798f7e3ebf9c2ac6c7c775f0b4_0_0.bam",filter_b=0)