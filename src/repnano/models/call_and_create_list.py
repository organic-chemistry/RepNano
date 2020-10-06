import pandas as pd
import glob
import os
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root_downloaded', type=str )
parser.add_argument('--root_bigf', type=str )
parser.add_argument('--output_dir', type=str )


args = parser.parse_args()



root_downloaded = args.root_downloaded + "/"
root_bigf = args.root_bigf + "/"
output_dir = args.output_dir + "/"

list_f = "list_files.txt"
list_p = "list_percents.txt"
list_f = pd.read_csv(os.path.join(root_downloaded,list_f),names=["f"])
list_p = pd.read_csv(os.path.join(root_downloaded,list_p),names=["p"])

cmd = []

os.makedirs(output_dir,exist_ok=True)
for f,p in zip(list_f.f,list_p.p[:]):

    input_bigf = root_bigf + f"{p}.h5"
    output_dir_calling_from_previous_percent=output_dir + f"/output_file_{p}.fa"
    cmd = f"python src/repnano/models/predict_simple.py {input_bigf} --bigf --output={output_dir_calling_from_previous_percent} --overlap 10 --percent"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

    output_dir_training_from_calling_percent = output_dir + f"percent_{p}.csv"
    cmd = f"python src/repnano/data/create_list_percent.py --input {input_bigf} --output {output_dir_training_from_calling_percent} --percent_file {output_dir_calling_from_previous_percent}_percentBrdu --plot --percent {p}"
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)
    #eventually compute initial percent from previous model



