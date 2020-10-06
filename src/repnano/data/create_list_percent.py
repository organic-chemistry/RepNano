"""
Takes a preprocessed file and generate a file which contaitn the
name of the biffile, read name and percent value, and a possible metadata

"""
import h5py
import pandas as pd
import argparse
import os
import numpy as np
import pylab
import matplotlib as mpl

mpl.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str )
parser.add_argument('--output', type=str)
parser.add_argument('--percent', type=float)
parser.add_argument('--type', type=str,help="Raw or Events",default="Events")
parser.add_argument('--metadata', type=str,default="")
parser.add_argument('--percent_file', type=str ,default="")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--threshold', type=float,default=0.04)


args = parser.parse_args()


#create output directory
p,_ = os.path.split(args.output)
os.makedirs(p,exist_ok=True)

target_percent_value = 0
if args.percent_file:
    nbin=101
    target_percent = pd.read_csv(args.percent_file,sep=" ",names=["readname","percent","error"])
    target_percent_sub = target_percent[target_percent.error < args.threshold]
    h,e = np.histogram(target_percent_sub.percent,bins=nbin,range=[0,1])
    m = np.max(h)
    p = np.argmax(h)
    width = 0
    #print(p)
    i=0
    for i in range(p,0,-1):
        if h[i]< m/4:
            break

    if i * 100/nbin > 6: #10 is 10 percent
        #Separable
        threshold = p/nbin /2
        n_high = np.sum(target_percent_sub.percent > threshold )
        p_high = n_high / len(target_percent_sub)
        target_percent_value = min(args.percent/p_high,100)

        print(f"threshold {threshold:.2f}, p high {p_high:.2f} , target p high {target_percent_value:.2f}")


        pylab.hist(np.array(target_percent.percent),range=[0,1],bins=nbin,
                   label=f"threshold {threshold:.2f}, p high {p_high:.2f} , target p high {target_percent_value:.2f}")
        pylab.plot([p/nbin/2,p/nbin/2],[0,m])
        pylab.legend()
        nf = args.output[:-4]+"_histo.png"
        print("Writing ",nf)
        pylab.savefig(nf)

    else:
        print("Not separable")


        pylab.hist(np.array(target_percent.percent), range=[0, 1],label="Not separable",bins=nbin)
        pylab.legend()
        nf = args.output[:-4] + "_histo.png"
        print("Writing ", nf)
        pylab.savefig(nf)



data = []
file_path = os.path.abspath(args.input)
h5 = h5py.File(file_path, "r")
#print(target_percent)
skip=0
if args.percent_file and not target_percent.readname[0].startswith("/"):
    #print("Skip")
    skip = 1
for v in h5.values():
    #print(v.name)
    if target_percent_value != 0:
        selec = target_percent[ target_percent.readname == v.name[skip:]]
        if len(selec) != 0:
            #print("Found")
            #print(selec)
            if np.array(selec.percent)[0] > threshold:
                percent_v = target_percent_value
                error = np.array(selec.error)[0]
            else:

                percent_v = 0
                error=np.array(selec.error)[0]
        else:
            percent_v = target_percent_value
            error = 0
    else:
        percent_v=args.percent
        error=0

    data.append({"file_name":file_path,"readname":v.name,"percent":percent_v,"type":args.type,"metadata":args.metadata,"error":error})

pd.DataFrame(data).to_csv(args.output,index=False)