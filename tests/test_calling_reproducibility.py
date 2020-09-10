"""Tests for the reproducibility of the calling process.
We used a preprocessed file and check whether the result
is identical or consistant


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
from test_global_pipeline import compute_differences

@pytest.fixture(scope="session")
def run_calling():
    shutil.rmtree('tests/current_output_calling',ignore_errors=True)
    os.mkdir('tests/current_output_calling')
    process = subprocess.Popen("./tests/execute_pipeline_calling.sh", shell=True, stdout=subprocess.PIPE)
    process.wait()
    #print("Doing something")
    #raise
    return



def test_find_expected_files_calling(run_calling):
    root = "tests/current_output_calling"
    list_files = ["output_file.fa","output_file.fa_ratio_B"]
    for file_name in list_files:
        assert(os.path.exists(os.path.join(root,file_name)))


def test_compare_file_calling(run_calling):
    root = "tests/current_output_calling"
    ref = "tests/test_whole_pipeline_reference"
    list_files = ["output_file.fa","output_file.fa_ratio_B"]
    for file_name in list_files:
        f_ref = os.path.join(ref,file_name)
        f_new = os.path.join(root, file_name)

        numerical = False
        if "ratio" in f_ref:
            numerical = True

        if not numerical:
            #TM method is not stochastic
            assert (filecmp.cmp(f_ref, f_new, shallow=False))


        delta = compute_differences(f_ref,f_new,numerical=numerical)
        if "lines" in delta.keys():
            print(f"{f_ref} and {f_new} have different number of lines")
            assert(False)
        elif delta["letters"] > 100:
            print(f"{f_ref} and {f_new} have same number of lines")
            print(f"But are two different, hamming distance is {delta['letters']}")
            assert(False)
        elif "val" in delta.keys() and np.mean(delta["val"])>0.01:
            print(f"{f_ref} and {f_new} have same number of lines")
            print(f"But are two different, mean val is {np.mean(delta['val'])}")
            assert(False)