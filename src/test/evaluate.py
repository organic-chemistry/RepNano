from ..models.predict_model import process
import os
import subprocess
weights = "data/cluster/training/v9p5delta10-bis/my_model_weights-940.h5"
# weights = "data/cluster/training/v9p5delta10-new-weight-longer/my_model_weights-60.h5"  # a lot of B
#weights = "data/cluster/training//v9p5-delta10-oversamplingL/my_model_weights-190.h5"
#weights = "data/cluster/training//v9p5-delta10-ref-from-file-only-T/my_model_weights-470.h5"
#weights = "data/cluster/training//v9p5-delta10-ref-from-file/my_model_weights-50.h5"

#weights = "data/cluster/training/v9p5-delta10-oversamplingB/my_model_weights-20.h5"
weights = "data/cluster/training/v9p5-delta10-ref-from-file-bis-max-files/my_model_weights-9300.h5"


basename = "results/v9p5-best-B-20170908-R9.5-newchem/"

ref = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"
redo = 0
# Evaluate all the sample
list_dir = [["substituted", "sub_template"], ["control", "control_template"],
            ["control-k47211", "control-k47211_template"]]

list_dir = [["20170908-R9.5/AG-Thy/0", "20170908-R9.5/BTF_AG_ONT_1_FAH14273_A-select_pass"],
            ["20170908-R9.5/AH-BrdU/0", "20170908-R9.5/BTF_AH_ONT_1_FAH14319_A-select_pass"],
            ["20170908-R9.5/AI-CldU/0/", "20170908-R9.5/BTF_AI_ONT_1_FAH14242_A-select_pass"],
            ["20170908-R9.5/AK-EdU/0/", "20170908-R9.5/BTF_AK_ONT_1_FAH14211_A-select_pass"],
            ["20170908-R9.5/AL-IdU/0/", "20170908-R9.5/BTF_AL_ONT_1_FAH14352_A-select_pass"]]
for dire, out in list_dir:  # + list_dir[4:]:
    if redo:
        process(weights, directory="data/raw/%s/" % dire,
                output="data/processed/{0}{1}.fasta".format(basename, out), Nbases=5, reads="",
                filter=None, already_detected=False, Nmax=None)
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
