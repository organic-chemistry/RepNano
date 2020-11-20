"""
Takes a preprocessed file and generate a file which contaitn the
name of the biffile, read name and percent value, and a possible metadata

"""
from scipy.optimize import curve_fit


def get_target_percent(percent_file, g_percent_value,nbin=101):
    #print(percent_file)
    target_percent = pd.read_csv(percent_file, sep=" ", names=["readname", "percent", "error", "mod"])
    target_percent.readname = [standardize_name(name) for name in target_percent.readname]
    target_percent.percent /= 100

    target_percent_sub = target_percent[target_percent.error < args.threshold]
    print("Nselected",len(target_percent_sub))
    #print(target_percent_sub)
    h, e = np.histogram(target_percent_sub.percent, bins=nbin, range=[0, 1],density=True)
    base = target_percent["mod"][0]
    m = np.max(h)
    p = np.argmax(h)
    print(m,p)
    width = 0
    # print(p)
    i = 0
    for i in range(p, 0, -1):
        if h[i] < m / 4:
            break
    separable = False

    pylab.clf()

    try:
        def gaus(x, a, x0, sigma):
            return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        def two_gauss(x,a1,x01,sigma1,a2,x02,sigma2):
            if x01>x02 or x01< -0.1:
                return np.zeros_like(x) + 1000
            if not(0<sigma1<0.42) or not(0<sigma2<0.4) or a1 < 0 or a2 < 0:
                return np.zeros_like(x)+1000

            return gaus(x,a1,x01,sigma1) + gaus(x,a2,x02,sigma2)


        popt, pcov = curve_fit(two_gauss, e[:-1], h, p0=[m/2, 0.0, 0.1] + [m/2,g_percent_value / 100,0.1])

        #pylab.plot(e[:-1], h, 'b+:', label='data')

        #print(g_percent_value)
        print(popt)
        #print(pcov)
        error = np.mean((two_gauss(e[:-1], *popt)-h)**2)
        print("error",error)

        p = 100*popt[-2]

        if error < 0.40 and popt[-2] - popt[1] > 0.1:   # to account for the fact that histo is not normalised
            separable = True

        if error < 100:
            pylab.plot(e[:-1], two_gauss(e[:-1], *popt), 'ro:', label='fit')
    except:
        #fit error
        pass

    if not separable and i * 100 / nbin > 10:  # 10 is 10 percent
        separable=True

    threshold = p / nbin / 2
    if np.mean(target_percent_sub.percent > threshold) < 0.5:
        separable = False

    if separable:
        # Separable
        threshold = p / nbin / 2
        n_high = np.sum(target_percent_sub.percent > threshold)
        p_high = n_high / len(target_percent_sub)
        target_percent_value = min(g_percent_value / p_high, 100)

        print(f"threshold {threshold:.2f}, p read high {p_high:.2f} , target value high {target_percent_value:.2f}")
        pylab.hist(np.array(target_percent.percent), range=[0, 1], bins=nbin,density=True,
                   label=f"thres {threshold:.2f}, p read high {p_high:.2f} , target value high {target_percent_value:.2f}")
        pylab.plot([p / nbin / 2, p / nbin / 2], [0, m])
        pylab.legend()
        nf = args.output[:-4] + f"{base}_histo.png"
        print("Writing ", nf)
        pylab.savefig(nf)

    else:
        print("Not separable")


        pylab.hist(np.array(target_percent.percent), range=[0, 1], label="Not separable", bins=nbin,density=True)
        pylab.legend()
        nf = args.output[:-4] + f"{base}_histo.png"
        print("Writing ", nf)
        pylab.savefig(nf)
        target_percent_value = g_target
        threshold=0

    return target_percent, target_percent_value, threshold, base

if __name__ == "__main__":
    import h5py
    import pandas as pd
    import argparse
    import os
    import numpy as np
    import pylab
    import matplotlib as mpl
    from repnano.models.train_simple import iter_keys, get_type , standardize_name

    mpl.use("Agg")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str )
    parser.add_argument('--output', type=str)
    parser.add_argument('--percent', nargs="+",type=float)
    parser.add_argument('--type', type=str,help="Raw or Events",default="Events")
    parser.add_argument('--metadata', type=str,default="")
    parser.add_argument('--percent_file', nargs='+',type=str ,default=[""])
    parser.add_argument('--mods', nargs='+',type=str ,default=[""])

    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--threshold', type=float,default=0.04)



    args = parser.parse_args()


    #create output directory
    p,_ = os.path.split(args.output)
    os.makedirs(p,exist_ok=True)

    pf={}
    if args.percent_file != [""]:
        pf = {}
        for pfile,g_target,mod in zip(args.percent_file,args.percent,args.mods):
            pf[mod] = get_target_percent(pfile,g_target)
            #print(pf[mod][0])
            assert(mod==pf[mod][-1])


    assert(len(args.mods) == len(args.percent))

    data = []
    file_path = os.path.abspath(args.input)
    h5 = h5py.File(file_path, "r")
    #print(target_percent)
    skip=0
    typef = get_type(h5)
    #print(target_percent[:10])

    for read_name in iter_keys(h5,typef=typef):
        #print(read_name)
        if pf != {}:
            percent = []
            error = []
            for mod in args.mods:
                target_percent, target_percent_value, threshold,mod = pf[mod]
                selec = target_percent[ target_percent.readname == standardize_name(read_name)]
                if len(selec) != 0:
                    #print("Found",target_percent_value,threshold,np.array(selec.percent)[0])
                    #print(selec)
                    if np.array(selec.percent)[0] > threshold:
                        percent.append(target_percent_value)
                        error.append(np.array(selec.error)[0])
                    else:

                        percent.append(0)
                        error.append(np.array(selec.error)[0])
                else:
                    percent_v = target_percent_value
                    error.append(0)
        else:
            percent=args.percent
            error=[0] * len(percent)
        #print(percent)

        info = {"file_name":file_path,"readname":standardize_name(read_name),"type":args.type,"metadata":args.metadata}
        for mod,p,e in zip(args.mods,percent,error):
            info[f"percent_{mod}"] = p
            info[f"error_{mod}"] = e
        #break
        data.append(info)

    #np.savetxt(args.output, pd.DataFrame(data), delimiter=';')
    pd.DataFrame(data).to_csv(args.output,index=False)