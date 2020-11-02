import pandas as pd
import glob
import os
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root_downloaded', type=str )
parser.add_argument('--root_processed', type=str )
parser.add_argument('--ref', type=str )
parser.add_argument('--which', type=int,default=0 )

parser.add_argument('--output_dv',type=str)
parser.add_argument('--csv',type=str)




args = parser.parse_args()



root_downloaded = args.root_downloaded + "/"
#root = args.root_processed + "/"

list_f = "list_files.txt"
list_p = "list_percents.txt"
sub = "190205_MN19358_FAK41381_A"
list_f = pd.read_csv(os.path.join(root_downloaded,list_f),names=["f"])
list_p = pd.read_csv(os.path.join(root_downloaded,list_p),names=["p"])
ref="/home/jeanmichel/DATA/Reference_Genomes/S288C_R64-2-1/S288C_reference_sequence_R64-2-1_20150113.fa"
ref = args.ref
cmd = []
#output_dir_bf = root+"/files/"  # for big files
#output_dir_extracted = root+"/training_initial/"  # for extracted file

data = []
for ifile,(f,p) in enumerate(zip(list_f.f[:],list_p.p[:])):
    f = f[:-6]+"_fast5"

    start = root_downloaded+f+f"/env/ig/atelier/nanopore/cns/MN19358/RG_IN_PROCESS/{sub}/fast5_file_management/workdir/*"
    print(start)
    f5 = glob.glob(start)[0]
    f5 = glob.glob(f5+"/*")[0][:-7] + f"{args.which}.fast5"
    fq = root_downloaded+f[:-6]+".fastq"

    data.append({"key":"Brdu_%.2f"%p,"f5":f5,"fq":fq,"ref":ref,"percents":[p],"mods":["B"],"canonical":["T"],"long_name":["Brdu"],"mix":True})

    cmd = f"ln -s {f5}  {args.root_processed}/B_{ifile}_{os.path.split(f5)[1]}"
    print(cmd)
    #subprocess.run(cmd, shell=True, check=True)

pd.DataFrame(data).to_csv(args.csv,index=False)