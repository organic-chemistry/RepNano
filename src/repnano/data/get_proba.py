import numpy as np
import os, pickle
def nan_polate(A):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x = np.isnan(A).ravel().nonzero()[0]

    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A
def find_range(d):
    mini,maxi = np.percentile(d,[0.2,99.8])
    return mini,maxi
def find_common_range(d1,d2):
    m1,M1 = find_range(d1)
    m2,M2 = find_range(d2)
    return min(m1,m2),max(M1,M2)

def interpolate_for_dynamic_range_and_norm(histo,limit=1e-7):

    if np.any(histo)==0:
        histo[histo==0]=np.nan
        for extremity in [0,-1]:
            if np.isnan(histo[extremity]):
                histo[extremity]=limit
        histo=nan_polate(histo)
    histo[histo<limit]=limit
    return histo/np.sum(histo)

def predict_log_proba(v,ran,normed_histo,limit=1e-7):
    outsider=False
    #print(v,ran[0],ran[-1],int((v-ran[0])/(ran[1]-ran[0])),ran[1]-ran[0])
    bins = (ran[1]-ran[0])#/len(ran)
    binp = int((v-ran[0])/bins)
    if binp<0 or binp>len(normed_histo)-1:
        outsider=True
        return limit,outsider
    else:
        return np.log(normed_histo[binp]),outsider

def compute_histo(ref,compare=None,binh=50,selected_transitions=[]):
    ranges = []
    histo_ref=[]
    histo_compare = []
    for i in range(len(ref)):

        if compare == None:
            xr = find_range(ref[i])
        else:
            xr = find_common_range(ref[i], compare[i])

        histo,ran = np.histogram(ref[i],bins=binh,
                       range=xr,density=True);
        histo_ref.append(interpolate_for_dynamic_range_and_norm(histo))
        if (selected_transitions != []) and (i not in selected_transitions):
            histo_ref[-1]=np.ones_like(histo_ref[-1]*1e-7)
        ranges.append(ran)
        if compare is not None:
            hc,_ = np.histogram(compare[i], bins=binh,
                         range=xr, density=True);
            histo_compare.append(interpolate_for_dynamic_range_and_norm(hc))
            if (selected_transitions != []) and (i not in selected_transitions):
                histo_compare[-1] = np.ones_like(histo_compare[-1] * 1e-7)

    return ranges,histo_ref,histo_compare

from repnano.data.create_transition_matrix import get_signal_expected,\
    norm_median_unmodified,norm_median_unmodified,get_indexes, norm_mean_unmodified
def evaluate_dataset(list_reads,ranges,ref,compare=None,
                              existing_transition=None,
                              filtered=False, rescale=False,length=5,
                              norm=True,order=0):

    list_trans, d_trans = list_transition(length)

    errors = []
    probas = []
    for x in list_reads:


        if norm == "median_unmodified":
            signal = norm_median_unmodified(x, length)
        elif norm == "mean_unmodified":
            signal = norm_mean_unmodified(x, length)
        else:
            signal, _ = get_signal_expected(x, Tt=None, length=length)

        indexes = get_indexes(x,length=length)

        proba = []

        for pos_seq,(i1,isignal) in enumerate(zip(indexes[:], signal)):

            p1,out1 = predict_log_proba(isignal, ranges[i1], ref[i1], limit=1e-7)
            if compare is not None:
                p2,out2 = predict_log_proba(isignal, ranges[i1], compare[i1], limit=1e-7)
                proba.append([p1,out1,p2,out2,i1])
            else:
                proba.append([p1, out1,i1])

        #pad
        limit=1e-1
        if compare==None:
            temp = [limit,0,0]
        else:
            temp= [limit,0,limit,0,0]

        if length%2==1:
            for i in range(length//2):
                proba.insert(0,temp)
                proba.append(temp)
        else:
            #assign proba to leng//2+1
            proba.insert(0, temp)
            for i in range((length-1)//2):
                proba.insert(0,temp)
                proba.append(temp)
        probas.append(np.array(proba,dtype=np.float16))
    return probas
import pandas as pd
def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())

def write(list_reads,probas,global_th,enrichment_th,length,bed_f,motif,min_length=None):

    list_trans, d_trans = list_transition(length)

    selected = []
    delta_shift = length//2
    for seq, v in d_trans.items():
        if motif is None:
            selected.append(v)
        else:
            if len(motif) == 1:
                if motif in seq[delta_shift:delta_shift+1]:
                    selected.append(v)
            else:
                if motif in seq:
                    selected.append(v)
    #print(selected)
    def motif_in(proba, selected):
        inside = np.zeros_like(proba[::, 4].copy(), dtype=bool)
        for t in selected:
            inside[proba[::, 4] == t] = 1
        return inside

    valid_transition = []
    for read,proba in zip(list_reads,probas):

        if min is not None and len(proba) < min_length:
            continue
        motif_seq = motif_in(proba, selected=selected)
        mean_ref = np.mean((proba[::, 0] * (~motif_seq)))
        #print(mean_ref)
        #Remove read whose outside motif transitions are too far from the reference
        if (global_th is not None) and (mean_ref< global_th):
            continue
        #Smooth in order to take into account information from multiple transition
        delta=smooth(proba[::,2]-proba[::,0],4)

        if enrichment_th is None:
            relevant = motif_seq
        else:
            relevant = (delta * motif_seq) > enrichment_th
        meta = read["meta"]
        #print(meta)
        if meta["mapped_strand"] == "-":
            relevant = relevant[::-1]
            delta = delta[::-1]

        for transition in np.where(relevant)[0]:
            pos = meta["mapped_start"]+transition + delta_shift
            valid_transition.append([meta["id"],meta["mapped_chrom"],meta["mapped_strand"],
                                     pos,float(f"{delta[transition]:.2f}")])
    print(f"Writing to {bed_f}")
    pd.DataFrame(valid_transition,columns=["id","mapped_chrom","mapped_strand","pos","log"]).to_csv(bed_f,index=False)



if __name__ == "__main__":

    from repnano.data.create_transition_matrix import load_directory_or_file_or_transitions,\
        list_transition,load_dataset, get_base_middle, get_transition_matrix_ind, sort_by_delta_mean
    from repnano.data import create_transition_matrix
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', dest='ref', type=str, default=None)

    parser.add_argument('--prefix', dest='prefix', type=str, default="",
                        help="Prefix to put before the name of the matrices")

    parser.add_argument('--compare', dest='compare', type=str, default=None)
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)

    parser.add_argument('--length-window', dest='length', type=int, default=None)
    parser.add_argument('--max-number-of-reads', dest='max', type=int, default=None)
    parser.add_argument('--order', dest='order', type=int, default=0)

    parser.add_argument('--trim_border', dest='exclude', type=int, default=None,
                        help="Remove begining and end of the sequence")

    parser.add_argument('--create_only', dest='create_only',action="store_true",
                        help="Only create a ref matrix")


    parser.add_argument('--exclude_base', dest='base', default=None,
                        help="Base to exclude from the normalisation procedure")
    parser.add_argument('--norm_method', dest='norm', default="median_unmodified",
                        choices=["mean_unmodified","median_unmodified","fit_unmodified","nothing"])
    parser.add_argument('--show', action="store_true")
    parser.add_argument('--refine', action="store_true")
    parser.add_argument('--all-in-one',dest="all_in_one", action="store_true")

    parser.add_argument('--global_th', type=float, default=None)
    parser.add_argument('--enrichment_th', type=float, default=None)
    parser.add_argument('--output_bed', action="store_true")
    parser.add_argument('--motif', type=str,default=None)
    parser.add_argument('--min-length', dest='min', type=int, default=200)


    args = parser.parse_args()

    if args.norm == "fit_unmodified":
        raise f"norm {args.norm} not implemented (must pass ref mat)"

    if args.refine and not args.all_in_one:
        print("#########################")
        print("Warning ,refinement will be done using only the last file!!")


    load_ref, allready_computed_ref = load_directory_or_file_or_transitions(args.ref)
    load_compare, allready_computed_compare = load_directory_or_file_or_transitions(args.compare)
    list_trans, d_trans = list_transition(args.length)

    if allready_computed_compare is False:
        load_compare = [None,None]

    all_t = sort_by_delta_mean(load_ref[0],load_compare[0], args.length)

    ranges,histo_ref,histo_compare = compute_histo(load_ref[1],load_compare[1])
                                       #            selected_transitions = [d_trans[t[2]] for t in all_t[:50]])


    if args.base != None:
        def get_motif(x, length):
            #print("Modified")
            return get_base_middle(x, length, base=args.base)
    else:
        def get_motif(x, length):
            return np.zeros(len(x["mean"][:-length + 1]), dtype=bool)
    create_transition_matrix.get_motif = get_motif


    # Create directory to write probas:
    root_name = args.prefix
    name = f"{root_name}"
    dir = os.path.split(name)[0]

    if not os.path.exists(dir):
        os.makedirs(dir)

    tmp_list = load_directory_or_file_or_transitions(args.dataset)[0]
    if args.all_in_one:
        all_to_process = [tmp_list]
    else:
        all_to_process = [[f1] for f1 in tmp_list]

    for i, filel in enumerate(all_to_process):
        print("Processing",filel)
        data_0 = load_dataset(filel,args.max, exclude=args.exclude)

        probas = evaluate_dataset(data_0,ref=histo_ref,
                                            ranges=ranges,
                                            compare=histo_compare,
                                              length=args.length,
                                              norm=args.norm)



        with open(f"{root_name}probas_{i}.pick", "wb") as f:
            pickle.dump(probas, f)

        if args.output_bed:
            write(data_0, probas, global_th=args.global_th,enrichment_th=args.enrichment_th,
                  length=args.length,
                  bed_f=f"{root_name}probas_{i}.bed",motif=args.motif,
                  min_length=args.min)

    if args.show:
        import pylab
        pylab.figure(figsize=(20, 15))
        #print(len(probas))
        #print(probas[0].shape)
        #print(probas[0][:100])
        #print([np.sum(np.log(p[::,0])) for p in probas])
        refv = [np.mean(p[::,0]) for p in probas]

        pylab.hist(refv,alpha=0.7,label="ref")
        if load_compare is not None:
            compv = [np.mean(p[::, 2]) for p in probas]
            pylab.hist(compv,alpha=0.7,label="compare")
        #pylab.plot(ref_distribution)
            pylab.legend()
            pylab.show()
            pylab.figure()
            #refvscomp = np.array(refv)>np.array(compv)
            refvscomp = np.array([np.mean(p[::, 0]) -np.mean(p[::, 2]) for p in probas]) > 0
            #print(refvscomp)
            pylab.hist(np.array(refvscomp,dtype=np.int))
            pylab.xlabel("Proba Ref")
            #pylab.ylabel("Comp")
            pylab.show()

    if load_compare is not None and args.refine:

        #refv = [np.mean(np.log(p[::,0])) for p in probas]
        #compv = [np.mean(np.log(p[::, 2])) for p in probas]
        #refvscomp = np.array(refv) > np.array(compv)
        refvscomp = np.array([np.mean(np.log(p[::, 0]/p[::, 2])) for p in probas])>0
        sub_dataset = [data_0[i] for i,v in enumerate(~refvscomp) if v]
        print(len(data_0),len(sub_dataset))
        compare_mean, _, compare_distribution = get_transition_matrix_ind(sub_dataset,
                                                                          length=args.length,
                                                                          existing_transition=None,
                                                                          norm=args.norm)

        import os,pickle
        root_name = args.prefix
        name = f"{root_name}compare"
        dir = os.path.split(name)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(name, compare_mean)
        with open(f"{root_name}compare_distribution.pick","wb") as f:
            pickle.dump(compare_distribution,f)







