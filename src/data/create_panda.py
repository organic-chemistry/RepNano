import pandas as pd

import argparse
import json
from git import Repo
import os
from multiprocessing import Pool
import numpy as np


import glob

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str)
parser.add_argument('--root-directory', dest="root_directory", type=str)
parser.add_argument('--not-relative', dest='relative', action='store_false')
parser.add_argument('--replace', dest='replace', action='store_true')
parser.add_argument("--add-keys", dest="add_keys", type=str, nargs='*')
args = parser.parse_args()

"""
if os.path.exists(args.filename) and not args.replace:
    print("File already exists")
    exit()
"""
path, name = os.path.split(args.filename)
ls = glob.glob(path + "/" + args.root_directory + "/*")
print("found %i files" % len(ls))
if len(ls) == 0:
    print("Warning, root directory is relative to filename")
if os.path.exists(args.filename):
    datas = pd.read_csv(args.filename)
else:
    path, name = os.path.split(args.filename)
    ls = glob.glob(path + "/" + args.root_directory + "/*")
    ls = [args.root_directory + "/" + ils.split(args.root_directory)[1] for ils in ls]
    datas = pd.DataFrame({"filename": ls})

# print(args.add_keys)
if len(args.add_keys) % 2 != 0:
    print("should have key and value")
else:
    def convert(v):
        try:
            return float(v)
        except:
            return v

    for k, v in zip(args.add_keys[::2], args.add_keys[1:][::2]):
        if k in datas.columns:
            if not args.replace:
                print("Not replacing", k)
                continue
        datas[k] = [convert(v) for _ in range(len(datas))]

datas.to_csv(args.filename, index=False)
