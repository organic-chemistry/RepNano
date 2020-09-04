"""Tests for the whole pipeline which is in test.

Focuses on verifying if all the file are created and similar to
what is expected.

Run this from project root directory:
$ python -m pytest
"""

import pytest
import subprocess
import shutil
import os
import filecmp
import textdistance
import numpy as np

@pytest.fixture(scope="session")
def run_pipeline():
    shutil.rmtree('tests/current_output',ignore_errors=True)
    os.mkdir('tests/current_output')
    process = subprocess.Popen("./tests/execute_whole_pipeline.sh", shell=True, stdout=subprocess.PIPE)
    process.wait()
    #print("Doing something")
    #raise
    return



def test_find_expected_files_preprocess(run_pipeline):
    root = "tests/current_output"
    list_files = ["output.fast5"]
    for file_name in list_files:
        assert(os.path.exists(os.path.join(root,file_name)))


def test_find_expected_files_calling(run_pipeline):
    root = "tests/current_output"
    list_files = ["output_file.fa","output_file.fa_ratio_B"]
    for file_name in list_files:
        assert(os.path.exists(os.path.join(root,file_name)))


def compute_differences(f1_n,f2_n,numerical=False):
    with open(f1_n,"r") as f1 , open(f2_n,"r") as f2:
        lines_1 = f1.readlines()
        lines_2 = f2.readlines()
        if len(lines_1) != len(lines_2):
            return {"lines":len(lines_1)-len(lines_2),"n_line":len(lines_1)}
        d=0
        val = []
        #print(len(lines_2))
        for l1,l2 in zip(lines_1,lines_2):
            if l1 != l2:
                if not numerical:
                    d += textdistance.hamming(l1,l2)
                else:
                    if l1.startswith(">"):
                        d += textdistance.hamming(l1,l2)
                    else:
                        p1 = np.array(list(map(float,l1.strip().split())))
                        p2 = np.array(list(map(float,l2.strip().split())))
                        print(f"percent b string1 {np.nanmean(p1):.2f} percent b string 2 {np.nanmean(p2):.2f}")
                        val.append(np.abs(np.nanmean(p1)-np.nanmean(p2)))


    return  {"letters":d,"n_line":len(lines_1),"val":val}



def test_compare_file_calling(run_pipeline):
    root = "tests/current_output"
    ref = "tests/test_whole_pipeline_reference"
    list_files = ["output_file.fa","output_file.fa_ratio_B"]
    for file_name in list_files:
        f_ref = os.path.join(ref,file_name)
        f_new = os.path.join(root, file_name)
        numerical = False
        if "ratio" in f_ref:
            numerical = True
        delta = compute_differences(f_ref,f_new,numerical=numerical)
        if "lines" in delta.keys():
            print(f"{f_ref} and {f_new} have different number of lines")
            assert(False)
        elif delta["letters"] > 100:
            print(f"{f_ref} and {f_new} have same number of lines")
            print(f"But are two different, hamming distance is {delta['letters']}")
            assert(False)
        elif "val" in delta.keys() and np.mean(delta["val"])>0.10:
            print(f"{f_ref} and {f_new} have same number of lines")
            print(f"But are two different, mean val is {np.mean(delta['val'])}")
            assert(False)


def test_global_pipeline(run_pipeline):
    pass
"""
if __name__=="__main__":
    
    root = "tests/current_output"
    ref = "tests/test_whole_pipeline_reference"
    file_n ="/output_file.fa_ratio_B"
    delta = compute_differences(root+file_n,ref+file_n,numerical=True)
    print(delta)
    print(np.mean(delta["val"]))"""
