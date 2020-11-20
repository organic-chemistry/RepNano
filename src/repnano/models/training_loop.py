import subprocess

import argparse
import os
import json
import pandas as pd
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--training_info', type=str)
parser.add_argument('--not_do', dest="do",action="store_false")


args = parser.parse_args()


with open(args.training_info,"r") as f:
    training = json.loads("".join(f.readlines()))

trainig_repertory = training["training_repertory"]
os.makedirs(trainig_repertory,exist_ok=True)

dataset = pd.read_csv(args.dataset,sep=";")

for i in range(training["nloop"]):
    if i == 0:
        input_folder = training["initial_percent"]
    else:
        input_folder = output_folder

    model_folder = f"{trainig_repertory}/model_from_initial_loop{i}/"
    output_folder = f"{trainig_repertory}/training_from_initial_network_{i}/"
    add=""
    if i >= training["nloop"]-1:
        add = " --smalllr --lstm"

    percent_training = " ".join(training["training_key"])
    if type(training["validation_key"]) == float:
        percent_val = True
        percent_validation = training["validation_key"]
    else:
        percent_val = False
        percent_validation= " ".join(training["validation_key"])
    mods = ast.literal_eval(dataset["mods"][0])
    mods = ' '.join(mods)

    if not os.path.exists(model_folder):
        if "max_len" in training:
            m = training["max_len"]
            add += f" --max_len {m} "
        if i != 0:
            add += " --error"
        if not percent_val:
            add += f" --percents_validation {percent_validation} "
        else:
            add += f" --validation {percent_validation} "
        cmd = f"python src/repnano/models/train_simple.py --root_data  {input_folder} --root_save {model_folder} --percents_training {percent_training}  --mods {mods}" + add
        #cmd = f"python src/repnano/models/train_simple.py --root_data  {input_folder} --root_save {model_folder} --percents_training 0  --percents_validation 0 --error"
        print(cmd)
        if args.do:
            subprocess.run(cmd, shell=True, check=True)
    """
    try:
        cmd = f"python3 preprocess_dataset.py  --type dv --model {model_folder}/weights.hdf5 --output_dv {output_folder} --root_d /scratch/jarbona/data_Repnano/ --root_p data/preprocessed/ --ref /scratch/jarbona/repnanoV10/data/S288C_reference_sequence_R64-2-1_20150113.fa"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    except:
    """
    if not os.path.exists(output_folder):
        add=""
        if "max_len" in training:
            m = training["max_len"]
            add += f" --max_len {m}"
        cmd = f"python3 misc/preprocess_dataset.py  --type dv --model {model_folder}/weights.hdf5 --out {output_folder} --dataset {args.dataset} " + add
        print(cmd)
        if args.do:
            subprocess.run(cmd, shell=True, check=True)

