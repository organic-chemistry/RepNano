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
    return lt,{"".join(s):lt.index(s) for s in lt}



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
def get_transition_matrix_ind(list_reads, existing_transition=None, filtered=False, rescale=False,length=5,norm=True):

    Ttd = [[] for _ in range(4 ** length)]
    Plat = np.zeros((4 ** length))
    Np = np.zeros((4 ** length))

    list_trans, d_trans = list_transition(length)

    errors = []
    for x in list_reads:

        if existing_transition is not None:

            signal, Tm, th, res, error = get_rescaled_signal(x, TransitionM=existing_transition, filtered=filtered,length=length)
            errors.append(error)
        else:
            signal,_ = get_signal_expected(x, Tt=None,length=length)

            if norm == "median_unmodified":
                signal =  norm_median_unmodified(x,length)

        indexes = get_indexes(x,length=length)

        for i1,signal in zip(indexes[:], signal):

            Plat[i1] += signal
            Np[i1] += 1
            Ttd[i1].append(signal)

    #print(len(Plat),len(Ttd))
    for k,v in d_trans.items():
        Ttd[v] = np.array(Ttd[v],dtype=np.float16)

    return Plat / Np, errors,Ttd

def get_signal_expected(x, Tt,length=5):

    if (length % 2) == 0:
        signal = x["mean"][(length-1) // 2:-((length-1) // 2)]
        signal = signal[1:]-signal[:-1]

    else:
        signal = x["mean"][(length//2):-(length//2)]

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
    return is_T



def rescale_deltas(real, th, Tm):

    abs = np.abs(real-th)
    thres = np.percentile(abs, 50)
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


def get_rescaled_signal(x, TransitionM, filtered=False, rs={}, thresh=0.25,length=5):
    real, th = get_signal_expected(x, TransitionM,length=length)
    Tm = get_motif(x,length)
    #print(len(real),len(th),len(Tm),len(x["mean"]),len(get_indexes(x,length)))

    if rs == {}:
        # print("Comp")
        rs = rescale_deltas(real, th, Tm)

    new = real.copy()
    new = (new ) / rs["x"][1] + rs["x"][0]
    #print(rs["x"])
    """

    new = (real-np.median(real[~Tm])) /  stats.median_absolute_deviation(real[~Tm])
    #new = (real-np.median(real[~Tm]))/stats.median_absolute_deviation(real[~Tm])
    #new = new * stats.median_absolute_deviation(th[~Tm]) + np.mean(th[~Tm])


    # print(rs["x"])
    """
    """
    pylab.figure()
    pylab.plot(real[:200], "-o", label="old")
    pylab.plot(th[:200], "-o", label="th")
    pylab.plot(Tm[:200], "-o")
    pylab.legend()
    pylab.show()

    pylab.figure()
    pylab.plot(real[:200],"-o",label="Before norm")
    pylab.plot(new[:200],"-o",label="After norm")
    pylab.plot(Tm[:200],"-o")
    pylab.legend()
    pylab.show()

    pylab.figure()
    pylab.plot(new[:200],"-o",label="New")
    pylab.plot(th[:200],"-o",label="th")
    pylab.plot(Tm[:200],"-o")
    pylab.legend()
    pylab.show()
    """


    whole, NotT, T = deltas(new, th, Tm)
    #print(np.mean((real-th)**2),whole, NotT, T)
    # print(NotT)
    if filtered:
        if NotT > thresh:
            return [], [], [], [], NotT
    return new, Tm, th, rs, NotT

def load_dataset(files,maxf):
    from repnano.models.simple_utilities import load_events_bigf

    Nempty_short = 0
    reads = []

    for i, read in enumerate(files):

        X = [read]
        y = [[0, 0]]


        def fun(*args, **kwargs):
            return tqdm(load_events_bigf(*args, **kwargs))
        for val in fun(X, y, min_length=200,
                       raw=False, base=True,
                       maxf=maxf, verbose=False, extra=True):


            Xrt, yrt, fnt, extra_e = val


            if len(Xrt) == 0:
                Nempty_short += 1

                continue
            else:
                reads.append(Xrt[0])
        if maxf is not None and len(reads)> maxf:
            break
    return reads

def sort_by_delta_mean(TT,TB,length):
    trans = []
    list_trans, d_trans = list_transition(length)
    for k1,v1 in d_trans.items():
        trans.append([np.abs(TT[v1]-TB[v1]),TT[v1]-TB[v1],k1])
    trans.sort()
    return trans[::-1]

def sort_by_signicatively_different(Ttd,TtdB,length):
    trans = []
    list_trans, d_trans = list_transition(length)
    for k1,v1 in d_trans.items():
        _,p = stats.mannwhitneyu(Ttd[v1],TtdB[v1])
        trans.append([p,np.mean(Ttd[v1])-np.mean(TtdB[v1]),k1])
    trans.sort()
    return trans

def test_transitions(d_trans):
    x = {"bases": np.random.choice(["A", "T", "G", "C"], length * 4)}
    for pos, ind in enumerate(get_indexes(x, length)):
        k = "".join(x["bases"][pos:pos + length])  # ,d_trans[ind])
        # print(d_trans[k],ind)
        assert (d_trans[k] == ind)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref', dest='ref', type=str, default=None)
    parser.add_argument('--prefix', dest='prefix', type=str, default="",
                        help="Prefix to put befor the name of the matrices")

    parser.add_argument('--compare', dest='compare', type=str, default=None)
    parser.add_argument('--length-window', dest='length', type=int, default=None)
    parser.add_argument('--max-number-of-reads', dest='max', type=int, default=None)

    parser.add_argument('--exclude_base', dest='base', default=None,
                        help="Base to exclude from the normalisation procedure")
    parser.add_argument('--norm_method', dest='norm', default="median_unmodified",
                        choices=["median_unmodified","fit_unmodified"])
    parser.add_argument('--show', action="store_true")

    args = parser.parse_args()


    def load_directory_or_file_or_transitions(path):
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
            print("Found:")
            print(load)
        return load,all_ready_computed




    load_ref,allready_computed_ref = load_directory_or_file_or_transitions(args.ref)
    load_compare,allready_computed_compare = load_directory_or_file_or_transitions(args.compare)

    if not allready_computed_ref:
        data_0 = load_dataset(load_ref,args.max)
    else:
        ref_mean,ref_distribution  = load_ref
    if not allready_computed_compare:
        data_100 = load_dataset(load_compare,args.max)
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


    if norm == "median_unmodified":
        if not allready_computed_ref:
            ref_mean, _ , ref_distribution = get_transition_matrix_ind(data_0, length=length,norm="median_unmodified")
        if not allready_computed_compare:
            compare_mean, _ , compare_distribution = get_transition_matrix_ind(data_100, length=length,norm="median_unmodified")


    if norm == "fit_unmodified":

        if not allready_computed_ref:
            TTp=None
            for i in range(6):
                TT,  _, TtdT = get_transition_matrix_ind(data_0,length=length,existing_transition=TTp)
                if TTp is not None:
                    print(np.nanmean(TT[TT!=0]-TTp[TT!=0]),np.nanstd(TT[TT!=0]-TTp[TT!=0])/np.nanstd(TT[TT!=0]))
                    print(np.nanstd(TTp[TTp!=0]),np.nanstd(TT[TT!=0]))
                TTp=TT
            ref_mean = TT
            ref_distribution = TtdT
        if not allready_computed_compare:
            compare_mean, _ , compare_distribution = get_transition_matrix_ind(data_100,
                                                                           length=length,
                                                                           existing_transition=ref_mean)

    if args.show:
        pylab.figure(figsize=(20, 15))
        pylab.plot(ref_mean)
        pylab.plot(ref_distribution)
        pylab.show()


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
        np.save(f"{root_name}ref_{norm}",ref_mean)

        with open(f"{root_name}ref_distribution_{norm}.pick","wb") as f:
            pickle.dump(ref_distribution,f)

    if not allready_computed_compare:
        np.save(f"{root_name}compare_{norm}", compare_mean)
        with open(f"{root_name}compare_distribution_{norm}.pick","wb") as f:
            pickle.dump(compare_distribution,f)









