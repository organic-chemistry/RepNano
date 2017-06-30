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
for dire, out in list_dir:
    if redo:
        process(weights, directory="data/raw/%s/" % dire,
                output="data/processed/{0}{1}.fasta".format(basename, out), Nbases=5, reads="",
                filter="data/processed/%s.InDeepNano.test" % out)


# Split the training from the test
    exex = "python src/test/get_fasta_from_train-test.py data/processed/{0}{1}.fasta data/processed/{1}.InDeepNano.test data/processed/{0}{1}_test".format(
        basename, out)
    subprocess.call(exex, shell=True)

    exex = "bwa mem -x ont2d  {2}  data/processed/{0}{1}_test_T.fasta > data/processed/{0}{1}_test_T.sam".format(
        basename, out, ref)
    # print(exex)
    subprocess.call(exex, shell=True)

    exex = "python src/test/ExportStatAlnFromSamYeast.py data/processed/{0}{1}_test_T.sam".format(
        basename, out, ref)
    subprocess.call(exex, shell=True)
