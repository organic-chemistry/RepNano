import pandas as pd
import glob
import os
import subprocess
import argparse
import os
import json
import tempfile
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str )
parser.add_argument('--from_scratch', action="store_true")
parser.add_argument('--cmd_mega', type=str,default="")

parser.add_argument('--noplot', action="store_true")

parser.add_argument('--repnano_preprocess', action="store_true")
parser.add_argument('--max_len', type=int,default=None)

parser.add_argument('--type_model',default=["repnano","v6","dv"])
parser.add_argument('--model',type=str)
parser.add_argument('--out',type=str)
parser.add_argument('--final_size', type=int, default=100)
parser.add_argument('--njobs', type=int, default=4)
parser.add_argument('--transition_matrix', type=str, default=None)
parser.add_argument('--minimum_percent_highsample', type=float,default=0.5)

parser.add_argument('--only_one',action="store_true",
                    help="Process only one file even if a directory is given")
parser.add_argument('--keys', nargs='+', type=str,default=[],
                    help=" if specified perprocess only the given keys")

args = parser.parse_args()
"""
with open(args.training_file,"r") as f:
    training_file = json.loads("".join(f.readlines()))
"""

dataset = pd.read_csv(args.dataset,sep=";")
#print("data",dataset)
#print(dataset.columns)
out = args.out
#root = args.root_processed + "/"

cmd = []

os.makedirs(out,exist_ok=True)
print(out)
#exit()
preprocessed = []

for index, row in dataset.iterrows():

    f5 = row["f5"]
    if "fq" in dataset.columns:
        fq =row["fq"]
    ref = row["ref"]
    key = row["key"]

    if args.keys != [] and key not in args.keys:
        continue
    print(row["percents"])
    per =ast.literal_eval(row["percents"])
    per = ' '.join('%.2f'%p for p in per)
    mods =ast.literal_eval(row["mods"])
    mods = ' '.join(mods)
    cano =ast.literal_eval(row["canonical"])
    cano = ' '.join(cano)
    if "exclude" in dataset.columns:
        exclude = row["exclude"]
    else:
        exclude = ''

    mixture = row["mix"]
    if mixture == "True":
        mixture = True
    if mixture == "False":
        mixture = False

    #output_dir_data = os.makedirs(os.path.join(args.output_dir,key))
    maxl=""
    if args.max_len != None:
        maxl = f"--max_len {args.max_len}"
    if not args.noplot:
        add = " --plot"
    else:
        add = ""

    if args.from_scratch:
        out_dir = out + f"/preprocess/{key}/"
        os.makedirs(out_dir, exist_ok=True)
        if args.repnano_preprocess:
            out_file = f"{out_dir}/output.h5"
            if not os.path.isfile(f5):
                f5 = glob.glob(f5+"/*.fast5")
                f5.sort()
                f5 = f5[0]
            if f5.endswith("h5"):
                cmd = f"cp {f5} {out_dir}/output.h5"
            else:
                cmd = f"python src/repnano/data/preprocess.py  --hdf5 {f5} --fastq {fq} --ref {ref}  --output_name {out_file} --njobs {args.njobs} " + maxl
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
            preprocessed.append(out_file)
        else:
            out_file = out_dir+"/signal_mappings.hdf5"
            #first create tempory rep with
            if os.path.isfile(f5) or args.only_one:
                if not os.path.isfile(f5):
                    f5 = glob.glob(f5+"/*.fast5")
                    f5.sort()
                    f5 = f5[0]
                with tempfile.TemporaryDirectory() as tmpdirname:
                    print('created temporary directory', tmpdirname)
                    os.symlink(f5,tmpdirname+f"/{os.path.split(f5)[1]}")
                    cmd = f"megalodon {tmpdirname}  --output-directory {out_dir}  {args.cmd_mega}  --overwrite  --reference {ref}  "
                    print(cmd)
                    subprocess.run(cmd, shell=True, check=True)
            else:
                cmd = f"megalodon {f5}  --output-directory {out_dir}  {args.cmd_mega}  --overwrite  --reference {ref}  "
                print(cmd)
                subprocess.run(cmd, shell=True, check=True)
            preprocessed.append(out_file)

        out_file_percent = os.path.join(out,f"initial_percent/{key}/percent.csv")
        if not os.path.exists(out_file_percent):
            if 'exclude' in row:
                toadd=f"--exclude {row['exclude']}"
            else:
                toadd=""

            cmd = f"python src/repnano/data/create_list_percent.py  --input {out_file} --output  {out_file_percent} --percent {per} --mods {mods} "+toadd
            #print(cmd)
            subprocess.run(cmd, shell=True, check=True)
    else:
        preprocessed = row["preprocessed"]
        out_calling = os.path.join(out, f"{key}/output.fa")

        if args.type_model == "repnano":
            cmd = f"python src/repnano/models/predict_simple.py {preprocessed} --bigf --output={out_calling} --overlap 10 --percent"
        elif args.type_model == "v6":
            cmd = f"python src/repnano/models/predict_simple.py {preprocessed} --bigf --output={out_calling} --typem 1 --activation sigmoid --window-length 160 --overlap 10 --weight data/training/weights_filters-32kernel_size-3choice_pooling-pooling-Truepool_size-2neurones-100batch_size-50optimizer-adamactivation-sigmoidnc-1dropout-0bi-Falsecost-logcosh.hdf5 --percent"
        elif args.type_model == "dv":
            adddv = ""
            if args.transition_matrix != None:
                adddv += f" --transition_matrix {args.transition_matrix} "
            if exclude != '':
                adddv += f" --exclude {exclude} "
            cmd = f"python src/repnano/models/evaluate_simple_v2.py --file {preprocessed} --output {out_calling} --model {args.model} --percent " + maxl + add + f" --final_size {args.final_size} --mods {mods} --canonical {cano} {adddv}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

        out_percent = os.path.join(out, f"{key}/percent.csv")

        pf = ""
        for mod in ast.literal_eval(row["mods"]):
            pf += f"{out_calling}_percent_{mod} "
        if mixture:
            add += " --mixture"
        cmd = f"python src/repnano/data/create_list_percent.py --minimum_percent_highsample {args.minimum_percent_highsample} --input {preprocessed} --output {out_percent} --percent_file {pf} --mods {mods}  --percent {per}" + add
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    #eventually compute initial percent from previous model

if args.from_scratch:
    dataset["preprocessed"] = preprocessed
    dataset.to_csv(out + "/dataset_preprocessed.csv",index=False,sep=";")
