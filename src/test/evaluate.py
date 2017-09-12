from ..models.predict_model import process
import os
import subprocess
weights = "data/training/my_model_weights-2290.h5"

basename = "results/ctc_"

ref = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"
redo = 1
# Evaluate all the sample
list_dir = [["substituted", "sub_template"], ["control", "control_template"],
            ["control-k47211", "control-k47211_template"]]

list_dir = [["20170908-R9.5/AG-Thy/0", "20170908-R9.5/BTF_AG_ONT_1_FAH14273_A-select_pass"],
            ["20170908-R9.5/AH-BrdU/0", "20170908-R9.5/BTF_AH_ONT_1_FAH14319_A-select_fail.InDeepNano"]]
for dire, out in list_dir[:1]:
    if redo:
        process(weights, directory="data/raw/%s/" % dire,
                output="data/processed/{0}{1}.fasta".format(basename, out), Nbases=5, reads="",
                filter=None, already_detected=False, Nmax=10)
        # filter="data/processed/%s.InDeepNano.test" % outz , already_detected=False)

    exex = "python src/test/get_fasta_from_train-test.py data/processed/{0}{1}.fasta all data/processed/{0}{1}_test".format(
        basename, out)
    subprocess.call(exex, shell=True)

    exex = "bwa mem -x ont2d  {2}  data/processed/{0}{1}_test_T.fasta > data/processed/{0}{1}_test_T.sam".format(
        basename, out, ref)
    # print(exex)
    subprocess.call(exex, shell=True)

    exex = "python src/test/ExportStatAlnFromSamYeast.py data/processed/{0}{1}_test_T.sam".format(
        basename, out, ref)
    subprocess.call(exex, shell=True)

"""
for dire, out in list_dir[:2]:
    if redo:
        process(weights, directory="data/raw/%s/" % dire,
                output="data/processed/{0}{1}.fasta".format(basename, out), Nbases=5, reads="",
                filter="data/processed/%s.InDeepNano.test" % outz , already_detected=False)

    exex = "python src/test/get_fasta_from_train-test.py data/processed/{0}{1}.fasta data/processed/{1}.InDeepNano.test data/processed/{0}{1}_test".format(
        basename, out)
    subprocess.call(exex, shell=True)

    exex = "bwa mem -x ont2d  {2}  data/processed/{0}{1}_test_T.fasta > data/processed/{0}{1}_test_T.sam".format(
        basename, out, ref)
    # print(exex)
    subprocess.call(exex, shell=True)

    exex = "python src/test/ExportStatAlnFromSamYeast.py data/processed/{0}{1}_test_T.sam".format(
        basename, out, ref)
    subprocess.call(exex, shell=True)"""
