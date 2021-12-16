import itertools
import numpy as np
from scipy import optimize
from tqdm import tqdm
import pylab
from scipy import stats
import os
import glob
import pickle


def list_transition(length=5):
    lt = [list(s) for s in itertools.product(["A","T","C","G"], repeat=length)]
    return lt,{"".join(s):ind for ind,s in enumerate(lt)}



def get_indexes(x, length):
    number = np.zeros(len(x["bases"]), dtype=np.int)
    number[x["bases"] == "T"] = 1
    number[x["bases"] == "C"] = 2
    number[x["bases"] == "G"] = 3
    indexes = np.zeros((len(number) - length+1), dtype=np.int)
    for i in range(length):
        # print(len(indexes),len(number[i:len(number)-5+i]))
        indexes += number[i:len(number) - length  + i+1] * 4 ** (length - 1 - i)

    return indexes

def norm_median_unmodified(x,length):
    delta0, _ = get_signal_expected(x, Tt=None, length=length)
    Tm = get_motif(x, length)
    deltas = (delta0 - np.median(delta0[~Tm])) / stats.median_absolute_deviation(delta0[~Tm])
    return deltas

def norm_mean_unmodified(x,length):
    delta0, _ = get_signal_expected(x, Tt=None, length=length)
    Tm = get_motif(x, length)
    deltas = (delta0 - np.mean(delta0[~Tm])) / np.mean(delta0[~Tm]**2)
    return deltas
def get_transition_matrix_ind(list_reads,
                              existing_transition=None,
                              filtered=False, rescale=False,length=5,
                              norm=True,order=0,maxi=None):

    if maxi == None:
        average = np.sum([len(x["mean"]) for x in list_reads]) / 4 ** length
        maxi = int(15*average)
        print(f"Average number of point by monomer {average:.2}, setting maxi to: {maxi}")
    Ttd = [np.zeros(maxi,dtype=np.float16) for _ in range(4 ** length)]
    Plat = np.zeros((4 ** length))
    Np = np.zeros((4 ** length),dtype=np.int)

    list_trans, d_trans = list_transition(length)

    errors = []
    for x in tqdm(list_reads):
        error = False
        if existing_transition is not None:

            signal, Tm, th, res, error = get_rescaled_signal(x, TransitionM=existing_transition, filtered=filtered,length=length)
            errors.append(error)
        else:

            if norm == "median_unmodified":
                signal =  norm_median_unmodified(x,length)
            elif norm == "mean_unmodified":
                signal =  norm_mean_unmodified(x,length)
            else:
                signal, _ = get_signal_expected(x, Tt=None, length=length)
        if error:
            continue
        indexes = get_indexes(x,length=length)
        #print(signal.dtype)
        for pos_seq,(i1,isignal) in enumerate(zip(indexes[:], signal)):

            Plat[i1] += isignal

            if order == 0:
                if Np[i1]<maxi:
                    Ttd[i1][Np[i1]] = isignal

            if order == 1:
                if pos_seq >=1 and pos_seq < len(signal)-2:
                    Ttd[i1].append([signal[pos_seq-1:pos_seq+2]])
            Np[i1] += 1

    #print(len(Plat),len(Ttd))
    Plat = Plat / Np
    Saturated = 0
    for k,v in d_trans.items():
        Ttd[v] = Ttd[v][:min(Np[v],maxi-1)]
        if Np[v]> maxi-1:
            Saturated += 1
    print(f"Number of {length}-mer with more than {maxi} points: {Saturated}")
        #Plat[v] = np.median(Ttd[v])
    #Checking for nan or inf:
    if np.any(np.isinf(Plat)) or np.any(np.isnan(Plat)):
        raise "Probable saturation"
    return Plat, errors,Ttd

def get_signal_expected(x, Tt,length=5,extended=False):

    if not extended:
        if (length % 2) == 0:
            signal = x["mean"][(length-1) // 2:-((length-1) // 2)]
            signal = signal[1:]-signal[:-1]

        else:
            signal = x["mean"][(length//2):-(length//2)]
    else:
        if (length % 2) == 0:
            raise "not implemented"
        signal = x["mean"]

    if Tt is not None:
        indexes = get_indexes(x,length)
        th = Tt[indexes]

    else:
        th=None


    return signal, th

def get_base_middle(x, length,base="T"):
    if (length%2) ==1:
        is_T = (x["bases"][(length//2):-(length//2)] == base)
    elif (length %2) ==0:
        is_T = x["bases"][(length-1)//2:-((length-1)//2)] == base
        is_T = is_T[1:] | is_T[:-1]
    #print(base)
    #print(is_T)
    return is_T

def get_base_in(x, length,base="T"):
    is_T = np.zeros(len(x["bases"])-length+1,dtype=np.bool)
    for i in range(1,length-1):
        end = -length+1+i
        if end == 0:
            end = None
        is_T = is_T | (x["bases"][i:end] == base)
    return is_T

def rescale_deltas(real, th, Tm):

    abs = np.abs(real-th)
    thres = np.percentile(abs, 25)
    skip = Tm | (abs > thres)
    def f(x):
        #delta = (real[skip]  / x[1]  + x[0]- th[skip]) ** 2
        new = real[~skip]/x[1] +x[0]
        target = th[~skip]
        delta = (new - target)**2 # to set mean two 0
        delta2 = (np.mean(new**2)-np.mean(target**2))**2

        #delta[delta > np.percentile(delta, 50)] = 0
        return delta2 +np.mean(delta)#np.mean(delta2)#+np.mean(delta2)

    return optimize.minimize(f, [0, 1], method="Powell")



def deltas(which, th, Tm):
    return np.mean((which - th) ** 2), np.mean((which[~Tm] - th[~Tm]) ** 2), np.mean((which[Tm] - th[Tm]) ** 2)


def get_rescaled_signal(x, TransitionM,
                        filtered=False, rs={},
                        thresh=0.25,length=5,Tm=None,extended=False):

    real, th = get_signal_expected(x, TransitionM,length=length,
                                   extended=extended)

    if Tm is None:
        Tm = get_motif(x,length)
    #print(len(real),len(th),len(Tm),len(x["mean"]),len(get_indexes(x,length)))


    if rs == {}:
        # print("Comp")
        #rs = rescale_deltas(real, th, Tm)
        pass

    #new = real.copy()
    #new = (new ) / rs["x"][1] + rs["x"][0]

    if extended:
        Tmp = np.concatenate([np.array([True]*(length//2)),
                              Tm,
                              np.array([True] *( length // 2))])
    else:
        Tmp = Tm
    new = real.copy()
    new -= np.mean(new[~Tmp])
    newstd = np.std(new[~Tmp])


    if np.isnan(newstd) or np.isinf(newstd):
        newstd = np.std(new[~Tmp][:10000])
        #print(np.any(np.isnan(real)),np.any(np.isnan(new)),len(new))
    oldstd = np.std(th[~Tm])
    if np.isnan(oldstd) or np.isinf(oldstd):
        print("STd infinite skipping",np.mean(~Tm),len(Tm))
        oldstd = np.std(th[~Tm][:10000])
        if np.isnan(oldstd) or np.isinf(oldstd):
            error = True
            return [], [], [], [], error
        #print("WTF")

    #new = new* stats.median_absolute_deviation(th[~Tm]) / stats.median_absolute_deviation(new[~Tm]) + np.mean(th[~Tm])
    new = new * (oldstd/2+newstd/2) / newstd + np.mean(th[~Tm])
    if np.any(np.isnan(new)):
        print(oldstd,newstd,np.mean(th[~Tm]),np.mean(~Tm),len(Tm),np.mean(real) )
        print(x["bases"][:100])
        error = True
        return [], [], [], [], error

    #print(np.mean(new[~Tm]),np.mean(th[[~Tm]]),np.std(new[~Tm]),np.std(th[~Tm]))
    #print(rs["x"])
    """

    new = (real-np.median(real[~Tm])) /  stats.median_absolute_deviation(real[~Tm])
    #new = (real-np.median(real[~Tm]))/stats.median_absolute_deviation(real[~Tm])
    #new = new * stats.median_absolute_deviation(th[~Tm]) + np.mean(th[~Tm])


    # print(rs["x"])
    """

    #pylab.figure()
    #pylab.hist(real[:200], "-o", label="old")
    #pylab.plot(th[:200], "-o", label="th")
    #pylab.plot(Tm[:200], "-o")
    #pylab.legend()
    #pylab.show()
    """
    pylab.figure()
    rg=[-4,4]
    bins=100
    pylab.hist(real[:],label="Before norm",range=rg,bins=bins)
    pylab.hist(new[:],label="After norm",range=rg,bins=bins)
    pylab.legend()
    pylab.show()

    pylab.figure()
    pylab.hist(new[:],label="New",range=rg,bins=bins)
    pylab.hist(th[:],label="th",range=rg,bins=bins)
    pylab.legend()
    pylab.show()
    exit()

    """

    #whole, NotT, T = deltas(new, th, Tm)
    #print(np.mean((real-th)**2),whole, NotT, T)
    # print(NotT)
    #if filtered:
    #    if NotT > thresh:
    #        return [], [], [], [], NotT
    return new, Tm, th, rs, False#NotT

def load_dataset(files,maxf,exclude=None,min=200,chunk=None):
    from repnano.models.simple_utilities import load_events_bigf
    print("Trimming",exclude)
    Nempty_short = 0
    reads = []

    for i, read in enumerate(files):

        X = [read]
        y = [[0, 0]]


        def fun(*args, **kwargs):
            return tqdm(load_events_bigf(*args, **kwargs))
        for val in fun(X, y, min_length=min,
                       raw=False, base=True,
                       maxf=maxf, verbose=False, extra=True):


            Xrt, yrt, fnt, extra_e = val


            if len(Xrt) == 0:
                Nempty_short += 1

                continue
            else:
                read = Xrt[0]
                if exclude is not None:
                    read["mean"] = read["mean"][exclude:-exclude]
                    read["bases"] = read["bases"][exclude:-exclude]

                read["mean"] = np.array(read["mean"],dtype=np.float16)
                read["meta"] = extra_e[0][1]
                read["meta"]["id"] = fnt[0]
                if chunk is None:
                    reads.append(read)
                else:
                    for split in range(len(read["mean"])//chunk + 1):
                        nread = {}
                        nread["mean"] = read["mean"][split*chunk:(split+1)*chunk]
                        nread["bases"] = read["bases"][split*chunk:(split+1)*chunk]
                        if len(nread["mean"])< min:
                            continue
                        nread["meta"] = extra_e[0][1]
                        #Only start is needed for positionning
                        if nread["meta"]["mapped_strand"] == "+":
                            nread["meta"]["mapped_start"] = read["meta"]["mapped_start"] + split * chunk
                        else:
                            nread["meta"]["mapped_start"] = max(nread["meta"]["mapped_start"],
                                                                nread["meta"]["mapped_end"]-(split+1)*chunk)
                        nread["meta"]["id"] = fnt[0]
                        reads.append(nread)

            if maxf is not None and len(reads)> maxf:
                break
        #And exit the main loop
        if maxf is not None and len(reads) > maxf:
            break
    print("Nsequences",len(reads))
    return reads

def sort_by_delta_mean(TT,TB,length):
    trans = []
    list_trans, d_trans = list_transition(length)
    for k1,v1 in d_trans.items():
        if np.isnan(TT[v1]) or np.isnan(TB[v1]):
            continue
        trans.append([np.abs(TT[v1]-TB[v1]),TT[v1]-TB[v1],k1])
    trans.sort()
    return trans[::-1]

def sort_by_signicatively_different(Ttd,TtdB,length,which="mann"):
    trans = []
    list_trans, d_trans = list_transition(length)
    for k1,v1 in d_trans.items():
        if len(Ttd[v1]) == 0 or len(TtdB[v1]) == 0:
            continue
        if which == "mann":
            _,p = stats.mannwhitneyu(Ttd[v1],TtdB[v1])
        if which == "kolmogorov":
            _,p = stats.ks_2samp(Ttd[v1],TtdB[v1])

        trans.append([p,np.mean(Ttd[v1])-np.mean(TtdB[v1]),k1,len(Ttd[v1]),len(TtdB[v1])])
    trans.sort()
    return trans

def test_transitions(d_trans):
    x = {"bases": np.random.choice(["A", "T", "G", "C"], length * 4)}
    for pos, ind in enumerate(get_indexes(x, length)):
        k = "".join(x["bases"][pos:pos + length])  # ,d_trans[ind])
        # print(d_trans[k],ind)
        assert (d_trans[k] == ind)

def load_directory_or_file_or_transitions(path):
    if path is None:
        return [],False
    all_ready_computed = False
    if path.endswith(".pick"):
        with open(path,"rb") as f:
            distribution = pickle.load(f)
        mean = np.zeros(len(distribution))
        for iv,v in enumerate(distribution):
            mean[iv] = np.mean(v)
        all_ready_computed = True
        return [mean,distribution],all_ready_computed

    if os.path.isfile(path):
        load = [path]
    else:
        print("Looking for fast5 extension files")
        load = glob.glob(path + "/*.fast5")
        load.sort()
        if len(load) == 0:
            print("Looking for h5 extension files")
            load = glob.glob(path + "/*.h5")
            load.sort()
        print("Found:")
        print(load)
    return load,all_ready_computed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref', dest='ref', type=str, default=None)
    parser.add_argument('--prefix', dest='prefix', type=str, default="",
                        help="Prefix to put befor the name of the matrices")

    parser.add_argument('--compare', dest='compare', type=str, default=None)
    parser.add_argument('--length-window', dest='length', type=int, default=None)
    parser.add_argument('--max-number-of-reads', dest='max', type=int, default=None)
    parser.add_argument('--order', dest='order', type=int, default=0)
    parser.add_argument('--min-length', dest='min', type=int, default=200)

    parser.add_argument('--trim_border', dest='exclude', type=int, default=None,
                        help="Remove begining and end of the sequence")

    parser.add_argument('--chunk', dest='chunk', type=int, default=None,
                        help="Chunk sequence to obtain 'limilar sequence lengths'")

    parser.add_argument('--create_only', dest='create_only',action="store_true",
                        help="Only create a ref matrix")



    parser.add_argument('--exclude_base', dest='base', default=None,
                        help="Base to exclude from the normalisation procedure")
    parser.add_argument('--norm_method', dest='norm', default="median_unmodified",
                        choices=["mean_unmodified","median_unmodified","fit_unmodified","nothing"])
    parser.add_argument('--show', action="store_true")

    args = parser.parse_args()


    load_ref,allready_computed_ref = load_directory_or_file_or_transitions(args.ref)
    load_compare,allready_computed_compare = load_directory_or_file_or_transitions(args.compare)

    if not allready_computed_ref:
        data_0 = load_dataset(load_ref,args.max,exclude=args.exclude,min=args.min,chunk=args.chunk)
    else:
        ref_mean,ref_distribution  = load_ref
    if not allready_computed_compare:
        if args.compare is None and not args.create_only:
            #print("CHang")
            indexes  = np.array(np.arange(len(data_0)),dtype=np.int)
            np.random.shuffle(indexes)
            data_100 = np.array(data_0)[indexes[len(data_0)//2:]]
            data_0 = np.array(data_0)[indexes[:len(data_0)//2]]
        else:
            data_100 = load_dataset(load_compare,args.max,exclude=args.exclude,min=args.min,chunk=args.chunk)
    else:
        compare_mean,compare_distribution  = load_compare

    length = args.length
    norm=args.norm
    root_name=args.prefix

    # motif to exclude from normalisation
    # if noting to exclude must return false of length signal - length petamer + 1
    if args.base != None:
        def get_motif(x, length):
            return get_base_middle(x, length,base=args.base)
    else:
        def get_motif(x, length):
            return np.zeros(len(x["mean"][:-length+1]),dtype=bool)



    list_trans, d_trans = list_transition(length)
    test_transitions(d_trans)


    if norm in ["median_unmodified","nothing","mean_unmodified"]:
        print(norm)

        if not allready_computed_ref:
            ref_mean, _ , ref_distribution = get_transition_matrix_ind(data_0,
                                                                       length=length,
                                                                       norm=norm,
                                                                       order=args.order)
        if not allready_computed_compare and not args.create_only:
            compare_mean, _ , compare_distribution = get_transition_matrix_ind(data_100,
                                                                               length=length,
                                                                               norm=norm,
                                                                                order=args.order)


    if norm == "fit_unmodified":
        if not allready_computed_ref:
            TTp=None
            niter = 6
            TtdT = None
            for i in range(6):
                print(f"Iteration {i} over {niter}")
                if TtdT != None:
                    del TtdT
                    TtdT = None # In order to avoid to store it twice in memory
                TT,  _, TtdT = get_transition_matrix_ind(data_0,length=length,existing_transition=TTp,
                                                                       order=args.order)
                if TTp is not None:
                    #TT = TTp/2+TT/2
                    print(np.nanmean(TT[TT!=0]-TTp[TT!=0]),np.nanstd(TT[TT!=0]-TTp[TT!=0])/np.nanstd(TT[TT!=0]))
                    print(np.nanstd(TTp[TTp!=0]),np.nanstd(TT[TT!=0]))
                TTp=TT
            ref_mean = TT
            ref_distribution = TtdT
        if not allready_computed_compare and not args.create_only:
            compare_mean, _ , compare_distribution = get_transition_matrix_ind(data_100,
                                                                           length=length,
                                                                           existing_transition=ref_mean,
                                                                       order=args.order)

    if args.show:
        pylab.figure(figsize=(20, 15))
        pylab.plot(ref_mean)
        pylab.plot(ref_distribution)
        pylab.show()

    if not args.create_only:
        all_t = sort_by_delta_mean(ref_mean,compare_mean,length)
        print("Comparing",args.ref,args.compare)
        for transitions in all_t[:30]:
            print("%.2f %s" % (transitions[1], transitions[2]))

        significatively_different = sort_by_signicatively_different(ref_distribution,
                                                                    compare_distribution,length=length)

        print("Comparing",args.ref,args.compare)
        for transitions in significatively_different[:30]:
            print("%.2e %.2f %s"%(transitions[0],transitions[1],transitions[2]))

    if not allready_computed_ref:
        name = f"{root_name}ref"
        dir = os.path.split(name)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(name,ref_mean)

        with open(f"{root_name}ref_distribution.pick","wb") as f:
            pickle.dump(ref_distribution,f)

    if not allready_computed_compare and not args.create_only:
        name = f"{root_name}compare"
        dir = os.path.split(name)[0]
        if not os.path.exists(dir):
            os.makedirs(dir)
        np.save(name, compare_mean)
        with open(f"{root_name}compare_distribution.pick","wb") as f:
            pickle.dump(compare_distribution,f)









