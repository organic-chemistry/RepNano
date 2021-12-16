import pysam
import tqdm
import numpy as np
import re
import pandas as pd
def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())

def find1(stro, ch):
  # 0.100 seconds for 1MB str
  npbuf = np.frombuffer(bytes(stro,'utf-8'), dtype=np.uint8) # Reinterpret str as a char buffer
  return np.where(npbuf == ord(ch))[0]

def convert_to_coordinate_old(seq,Ml,Mm,which="T"):
    #Ml base proba
    #Mm number of base to skip
    #print(sum(Mm),sum(np.array(list(seq))=="T"))
    #print(Mm)
    #print(seq)
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


def convert_to_coordinate(seq, Ml, Mm, which=u"T"):
    # Ml base proba
    # Mm number of base to skip
    # print(sum(Mm),sum(np.array(list(seq))=="T"))
    # print(Mm)
    # print(seq)
    assert (len(Ml) == len(Mm))
    Mm = Mm + 1
    result = np.zeros((len(seq))) + np.nan
    n_which = 0
    # print(Mmc)
    cum = np.cumsum(Mm)
    cum -= 1
    #print(cum, cum.dtype)
    # which = r
    # array_s = np.fromiter(seq,dtype=np.char)
    # array_s=np.array(list(seq), dtype=np.unicode)
    #pos = np.array([m.start() for m in re.finditer(which, seq)])
    pos = find1(seq,which)
    # shold not append
    if len(cum) != 0:
        if cum[-1] > len(pos) - 1:
            # truncate
            cum = cum[:np.argmax(cum > len(pos) - 1)]
            Ml = Ml[:len(cum)]
        # print(pos)

        result[pos[cum]] = np.array(Ml) / 255

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

def load_read_bam(bam,filter_b=0.5,n_b=5000,verbose=False,fill_nan=False,res=1,
    maxi=None,chs=None,tqdm_do=False,calibration=False):
    """
    filter_b and n_b are here to select signal with Brdu
    it select the signal if n_b points are higher that filter_b
    If you want to keep everything you can set n_b to 0
    res is the resolution of the final signal
    for example at 100, it does a smoothing average over 100 points and then select one point every 100 points

    chs can be a list of chromosome that you want to keep
    for example ["chr1","chr2"]
    the nomenclature has to be the same as the one of your reference file

    maxi is the maximum number of read that you want to process

    it returns an array for each read with [attr,b_val]
    The x coordinate is computed as x = np.arange(len(b_val)) * res + attr["mapped_start"]
    If the strand mapped to is "-" I already flipped the signal.
    (it means that the x coordinate are always increasing)
    """
    #print(bam)
    samfile = pysam.AlignmentFile(bam, "r")#,check_sq=False)

    Read ={}

    monitor = lambda x : x
    if tqdm_do:
        monitor = lambda x: tqdm.tqdm(x)
    for ir,read in monitor(enumerate(samfile)):
        if verbose:
            print(ir,read)
        #seq,Ml,Mm = read.get_forward_sequence(),read.get_tag("Ml"),[int(v) for v in read.get_tag("Mm")[:-1].split(",")[1:]]
        seq = read.get_forward_sequence()

        #print(Mm2)
        #print(len(Mm),len(Mm2))
        attr={}
        if read.is_reverse:
            attr["mapped_strand"] = "-"
        else:
            attr["mapped_strand"] = "+"


        attr["mapped_chrom"] = read.reference_name


        pos = read.get_reference_positions()
        attr["mapped_start"] = pos[0]
        attr["mapped_end"] = pos[-1]
        attr["seq"]=seq
        #print(read.reference_name, read.reference_id)
        if chs is not None and attr["mapped_chrom"] not in chs:
            continue
        try:
            Ml = read.get_tag("Ml")
        except:
            Ml = np.array([])
            # print(read.get_tag("Mm"))
            # Mm = [int(v) for v in read.get_tag("Mm")[:-1].split(",")[1:]]
            # print(Mm)
        Mm = np.fromstring(read.get_tag("Mm")[4:-1], dtype=np.int, sep=',')
        #print(Mm)
        val=convert_to_coordinate(seq,Ml,Mm)
        if fill_nan:
            val[np.isnan(val) & (np.array(list(seq)) == "T")] = 0

        if res != 1:
            val = smooth(val,res)
            val = np.array(val[::res],dtype=np.float16)

        if attr["mapped_strand"] == "-":
            Nn = val[::-1]
        else:
            Nn = val

        if calibration:
            lg = lambda x: 1/(1+np.exp(-(x-0.16)*30))
            Nn = lg(Nn)

        if np.sum(np.array(Nn)>filter_b) >= n_b:
            Read[read.query_name] = [attr,Nn] #,get_longest_low(Nn)]

        if maxi is not None and ir >= maxi-1:
            break
        #break
    return Read


def load_read_bam_multi(bam,filter_b=0.5,n_b=5000,verbose=False,
                        fill_nan=False,res=1,maxi=None,chs=None,
                        tqdm_do=False,allready_mod=False):
    """
    filter_b and n_b are here to select signal with Brdu
    it select the signal if n_b points are higher that filter_b
    If you want to keep everything you can set n_b to 0
    res is the resolution of the final signal
    for example at 100, it does a smoothing average over 100 points and then select one point every 100 points

    chs can be a list of chromosome that you want to keep
    for example ["chr1","chr2"]
    the nomenclature has to be the same as the one of your reference file

    maxi is the maximum number of read that you want to process

    it returns an array for each read with [attr,b_val]
    The x coordinate is computed as x = np.arange(len(b_val)) * res + attr["mapped_start"]
    If the strand mapped to is "-" I already flipped the signal.
    (it means that the x coordinate are always increasing)
    """
    #print(bam)
    if fill_nan:
        print("Non implemented")
    samfile = pysam.AlignmentFile(bam, "r")#,check_sq=False)

    Read ={}

    monitor = lambda x : x
    if tqdm_do:
        monitor = lambda x: tqdm.tqdm(x)

    for ir,read in monitor(enumerate(samfile)):
        if verbose:
            print(ir,read)
        #seq,Ml,Mm = read.get_forward_sequence(),read.get_tag("Ml"),[int(v) for v in read.get_tag("Mm")[:-1].split(",")[1:]]
        seq = read.get_forward_sequence()

        #print(Mm2)
        #print(len(Mm),len(Mm2))
        attr={}
        if read.is_reverse:
            attr["mapped_strand"] = "-"
        else:
            attr["mapped_strand"] = "+"


        attr["mapped_chrom"] = read.reference_name


        pos = read.get_reference_positions()
        attr["mapped_start"] = pos[0]
        attr["mapped_end"] = pos[-1]
        attr["seq"]=seq
        #print(read.reference_name, read.reference_id)
        #print(read.header)
        #for attrn in dir(read):
    #        print(attrn,getattr(read,attrn))


        if chs is not None and attr["mapped_chrom"] not in chs:
            continue

        try:
            Ml = read.get_tag("Ml")
        except:
            Ml = np.array([])


        Mmt = read.get_tag("Mm").split(";")[:-1]
        #print(Mmt)
        Mm = {}
        base_ref={}
        for Smm in Mmt:
            base = Smm[2:3]
            #shift = [int(v) for v in Smm.split(",")[1:]]
            shift= np.fromstring(Smm[4:], dtype=np.int, sep=',')
            #print(Smm[:3])
            #Mm = np.fromstring(read.get_tag("Mm")[4:-1], dtype=np.int, sep=',')
            Mm[base]=shift
            base_ref[base]=Smm[:1]
        #print(Mm)
        if Mm != {}:
            pass
            # print(read.get_tag("Mm"))
            # Mm = [int(v) for v in read.get_tag("Mm")[:-1].split(",")[1:]]
            # print(Mm)
        #print(Mm)
        Nn ={}
        start = 0
        for mod in Mm.keys():

            val = convert_to_coordinate(seq,Ml[start:start+len(Mm[mod])],Mm[mod],which=base_ref[mod])
            #val[np.isnan(val) & (np.array(list(seq))=="T")]=0
            start += len(Mm[mod])

            if res != 1:
                val = smooth(val,res)
                val = np.array(val[::res],dtype=np.float16)

            if attr["mapped_strand"] == "-":
                val = val[::-1]


            if not allready_mod:
                val[np.isnan(val) & (np.array(list(seq)) == base_ref[mod])] = 0
            Nn[mod]=val


        #if np.sum(np.array(Nn[mod])>filter_b) >= n_b:
        Read[read.query_name] = [attr,Nn] #,get_longest_low(Nn)]

        if maxi is not None and ir >= maxi-1:
            break
        #break
    return Read


#load_read_bam("../../../debug/debug_GPU_ref/bam_runid_1ca9cbe0f2c9d4798f7e3ebf9c2ac6c7c775f0b4_0_0.bam",filter_b=0)
