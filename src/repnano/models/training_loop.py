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
parser.add_argument('--noerror', dest="error",action="store_false")
parser.add_argument('--restart', dest="restart",action="store_true")
parser.add_argument('--training_repertory',default=None)
parser.add_argument('--root_training_dataset',default=None)


args = parser.parse_args()


with open(args.training_info,"r") as f:
    training = json.loads("".join(f.readlines()))

training_repertory = training.get("training_repertory",args.training_repertory)
if training_repertory == None:
    print("training repertory must be defined")
    raise
if args.training_repertory != None and training_repertory != args.training_repertory:
    print("Two different rep defined (one in json, one in command line)")
    raise

os.makedirs(training_repertory,exist_ok=True)

dataset = pd.read_csv(args.dataset,sep=";")

for i in range(training["nloop"]):
    if i == 0:
        input_folder = args.root_training_dataset
    else:
        input_folder = output_folder

    model_folder = f"{training_repertory}/model_from_initial_loop{i}/"
    output_folder = f"{training_repertory}/training_from_initial_network_{i}/"
    add=""
    if i == training["nloop"]-1:
        add = " --lstm"
    elif i == training["nloop"]-2:
        add = " --lstm "


    percent_training = " ".join(training["training_key"])
    if type(training["validation_key"]) == float:
        percent_val = True
        percent_validation = training["validation_key"]
    else:
        percent_val = False
        percent_validation= " ".join(training["validation_key"])
    mods = ast.literal_eval(dataset["mods"][0])
    mods = ' '.join(mods)

    if training.get("transition_matrix","") != "":
        add += f" --transition_matrix {training['transition_matrix']}"

    if not os.path.exists(model_folder) or (args.restart and i==training["nloop"]-1):
        if "max_len" in training:
            m = training["max_len"]
            add += f" --max_len {m} "
        if i >= training["nloop"]-1 and args.error:
            add += " --error"
        if args.restart:
            add += f"--weights {model_folder}/weights.hdf5"
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
        if training.get("transition_matrix", "") != "":
            add += f" --transition_matrix {training['transition_matrix']}"
        cmd = f"python3 misc/preprocess_dataset.py --minimum_percent_highsample {training.get('minimum_percent_highsample',0.5)} --type dv --model {model_folder}/weights.hdf5 --out {output_folder} --dataset {args.dataset} " + add
        print(cmd)
        if args.do:
            subprocess.run(cmd, shell=True, check=True)
