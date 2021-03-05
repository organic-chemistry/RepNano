import itertools
import numpy as np
from scipy import optimize
from tqdm import tqdm
from repnano.models.simple_utilities import load_events_bigf
import pylab
from scipy import stats


def list_transition(length=5):
    lt = [list(s) for s in itertools.product(["A","T","C","G"], repeat=length)]
    return lt,{"".join(s):lt.index(s) for s in lt}

#Original code but does not seems to work...
"""
def get_indexes(x):
    number = np.zeros(len(x["bases"]),dtype=np.int)
    number[x["bases"]=="T"]=1
    number[x["bases"]=="C"]=2
    number[x["bases"]=="G"]=3
    indexes = np.zeros((len(number)-4),dtype=np.int)
    for i in range(5):
        #print(len(indexes),len(number[i:len(number)-5+i]))
        indexes += number[i:len(number)-4+i]*4**i

"""


def get_indexes(x, length=5):
    number = np.zeros(len(x["bases"]), dtype=np.int)
    number[x["bases"] == "T"] = 1
    number[x["bases"] == "C"] = 2
    number[x["bases"] == "G"] = 3
    indexes = np.zeros((len(number) - length + 1), dtype=np.int)
    for i in range(length):
        # print(len(indexes),len(number[i:len(number)-5+i]))
        indexes += number[i:len(number) - length + 1 + i] * 4 ** (length - 1 - i)

    return indexes

def norm_median_unmodified(x,length):
    delta0, _ = get_signal_expected_ind(x, Tt=None, length=length)
    Tm = get_motif(x, length)
    deltas = (delta0 - np.median(delta0[~Tm])) / stats.median_absolute_deviation(delta0[~Tm])
    return deltas
def get_transition_matrix_ind(list_reads, existing_transition=None, filtered=False, rescale=False,length=5,norm=True):
    Tt = np.zeros((4 ** length, 4 ** length))
    Ttd = {}
    Plat = np.zeros((4 ** length))

    list_trans, d_trans = list_transition(length)


    N = np.zeros((len(list_trans), len(list_trans)))
    Np = np.zeros((4 ** length))
    errors = []
    for x in list_reads:

        if existing_transition is not None:

            plateaus = []
            deltas, Tm, th, res, error = get_rescaled_deltas(x, TransitionM=existing_transition, filtered=filtered,length=length)
            errors.append(error)
        else:
            plateaus = x["mean"][2:-2]
            deltas,_ = get_signal_expected_ind(x, Tt=None,length=length)

            if norm == "median_unmodified":
                deltas =  norm_median_unmodified(x,length)

        indexes = get_indexes(x)
        globalc = False
        if globalc:
            Tt[indexes[:-1], indexes[1:]] += deltas
            N[indexes[:-1], indexes[1:]] += 1
            if len(plateaus) != 0:
                Plat[indexes] += plateaus
                Np[indexes] += 1
        else:
            if len(plateaus) != 0:
                for i1, i2, delta, plat in zip(indexes[:-1], indexes[1:], deltas, plateaus[:-1]):
                    Tt[i1, i2] += delta
                    N[i1, i2] += 1

                    Plat[i1] += plat
                    Np[i1] += 1
                    key = "%s,%s" % (str(i1), str(i2))
                    if key not in Ttd:
                        Ttd[key] = [delta]
                    else:
                        Ttd[key].append(delta)
            else:
                for i1, i2, delta in zip(indexes[:-1], indexes[1:], deltas):
                    Tt[i1, i2] += delta
                    N[i1, i2] += 1

                    key = "%s,%s" % (str(i1), str(i2))
                    if key not in Ttd:
                        Ttd[key] = [delta]
                    else:
                        Ttd[key].append(delta)

    # print(np.sum(N),len(deltas))



    Tt2 = np.zeros_like(Tt)


    N[N == 0] = 1
    Np[Np == 0] = 1
    Tt /= N
    """
    for k,val in Ttd.items():
        i1,i2=[int(v) for v in k.split(",")]
        Tt[i1,i2]=np.median(val)
    """
        #pylab.hist(val,label=f"{np.median(val)} {np.mean(val)}",bins=80)
        #pylab.plot([np.median(val),np.median(val)],[0,100,],"-o")
        #pylab.legend()
        #pylab.show()
        #exit()

    if rescale and existing_transition is not None:
        om = np.mean(existing_transition[existing_transition != 0])
        stdo = np.std(existing_transition[existing_transition != 0])
        Tt[Tt != 0] = stdo * Tt[Tt != 0] / np.std(Tt[Tt != 0])
        Tt[Tt != 0] = (Tt[Tt != 0] - np.mean(Tt[Tt != 0])) + om
    return Tt, Plat / Np, errors, Tt2, Ttd

def get_signal_expected_ind(x, Tt,length=5):

    delta = x["mean"][1:] - x["mean"][:-1]
    delta= delta[length - 3:-2]
    if Tt is not None:
        indexes = get_indexes(x,length)
        th = Tt[indexes[:-1],indexes[1:]]
    else:
        th=None

    return delta, th

def get_tmiddle(x, length):
    Tm = []
    for n in range(length-3, len(x["bases"]) - 3):
        if x["bases"][n] == "T" or x["bases"][n + 1] == "T":
            Tm.append(True)
        else:
            Tm.append(False)
    return np.array(Tm)

def get_motif(x,length):
    return get_tmiddle(x,length)

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


def get_rescaled_deltas(x, TransitionM, filtered=False, rs={}, thresh=0.25,length=5):
    real, th = get_signal_expected_ind(x, TransitionM)
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
            if len(reads)> maxf:
                break
    return reads

def sort_by_delta_mean(TT,TB,length):
    trans = []
    list_trans, d_trans = list_transition(length)
    for k1,v1 in d_trans.items():
        for k2, v2 in d_trans.items():

            if np.abs(TT[v1,v2]) > 1e-7:
                trans.append([np.abs(TT[v1,v2]-TB[v1,v2]),TT[v1,v2]-TB[v1,v2],k1,k2])
    trans.sort()
    return trans[::-1]

def sort_by_signicatively_different(Ttd,TtdB,length):
    trans = []
    list_trans, d_trans = list_transition(length)
    for k1,v1 in d_trans.items():
        for k2, v2 in d_trans.items():

            if np.abs(TT[v1,v2]) > 1e-7:
                key="%i,%i"%(v1,v2)
                _,p = stats.mannwhitneyu(Ttd[key],TtdB[key])
                trans.append([p,TT[v1,v2]-TB[v1,v2],k1,k2])
    trans.sort()
    return trans


if __name__ == "__main__":


    root_name="10"
    data_0 = load_dataset(["test_training_mat/Brdu_0.h5"],1000)
    data_100 = load_dataset(["test_training_mat/Brdu_10.h5"],1000)

    norm="median_unmodified"
    #norm="fit_unmodified"


    length = 5
    list_trans, d_trans = list_transition(length)

    #Small test
    x = {"bases": np.random.choice(["A", "T", "G", "C"], length * 4)}
    for pos, ind in enumerate(get_indexes(x, length)):
        k = "".join(x["bases"][pos:pos + length])  # ,d_trans[ind])
        # print(d_trans[k],ind)
        assert (d_trans[k] == ind)

    if norm == "median_unmodified":
        TT, PlatT, _, Tt2, TtdT = get_transition_matrix_ind(data_0, length=length,norm="median_unmodified")
        TB, PlatT, _, Tt2, TtdB = get_transition_matrix_ind(data_100, length=length,norm="median_unmodified")



    if norm == "fit_unmodified":
        TTp=None

        for i in range(6):
            TT, PlatT, _, Tt2, TtdT = get_transition_matrix_ind(data_0,length=length,existing_transition=TTp)
            if TTp is not None:
                print(np.nanmean(TT[TT!=0]-TTp[TT!=0]),np.nanstd(TT[TT!=0]-TTp[TT!=0])/np.nanstd(TT[TT!=0]))
                print(np.nanstd(TTp[TTp!=0]),np.nanstd(TT[TT!=0]))
            TTp=TT

        TB, PlatT, _, Tt2, TtdB = get_transition_matrix_ind(data_100,length=length,existing_transition=TT)


    pylab.figure(figsize=(20, 15))
    pylab.plot(TT[TT != 0].flatten())
    pylab.plot(TB[TT != 0].flatten())
    pylab.show()


    all_t = sort_by_delta_mean(TT,TB,length)
    print("TT1,TB")
    for transitions in all_t[:30]:
        print(transitions[1:])

    significatively_different = sort_by_signicatively_different(TtdT,TtdB,length=length)

    print("TT1,TB")
    for transitions in significatively_different[:30]:
        print(transitions[:])

    np.save(f"{root_name}TT_{norm}",TT)
    np.save(f"{root_name}TB_{norm}",TB)

    import pickle
    with open(f"{root_name}Td_{norm}.pick","wb") as f:
        pickle.dump(TtdT,f)

    with open(f"{root_name}Bd_{norm}.pick","wb") as f:
        pickle.dump(TtdB,f)









