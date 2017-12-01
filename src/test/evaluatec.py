from ..models.predict_model import process
import os
import subprocess


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Nbases', type=int, choices=[4, 5, 8], default=4)
parser.add_argument('--root', type=str, default="data/results/")
parser.add_argument('--size', type=int, default=40)
parser.add_argument('--sample', type=int, default=200)

parser.add_argument('--weight', dest='weight', type=str, default=None)
parser.add_argument('--clean', dest="clean", action="store_true")
parser.add_argument('--attention', dest="attention", action="store_true")
parser.add_argument('--residual', dest="res", action="store_true")


args = parser.parse_args()


S = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1))
KTF.set_session(S)


weights = args.weights
basename = args.root


ref = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"
redo = 1
# ref = "data/external/chromFa/*.fa"
# redo = 0

# Evaluate all the sample
list_dir = [["substituted", "sub_template", 5], ["control", "control_template", 5],
            ["control-k47211", "control-k47211_template", 5]]

list_dir = [["20170908-R9.5/AB-2minBrdU", "20170908-R9.5/prout_2", 5],
            ["20170908-R9.5/AD-basecalled", "20170908-R9.5/prout", 5],
            ["20170908-R9.5/AG-basecalled", "20170908-R9.5/BTF_AG_ONT_1_FAH14273_A-select_pass", 8],
            ["20170908-R9.5/AH-basecalled", "20170908-R9.5/BTF_AH_ONT_1_FAH14319_A-select_pass", 5],
            ["20170908-R9.5/AG-Thy/", "20170908-R9.5/BTF_AG_ONT_1_FAH14273_A-select_pass", 5],
            ["20170908-R9.5/AH-BrdU/", "20170908-R9.5/BTF_AH_ONT_1_FAH14319_A-select_pass", 5],
            ["20170908-R9.5/AI-CldU/0/", "20170908-R9.5/BTF_AI_ONT_1_FAH14242_A-select_pass", 5],
            ["20170908-R9.5/AK-EdU/0/", "20170908-R9.5/BTF_AK_ONT_1_FAH14211_A-select_pass", 5],
            ["20170908-R9.5/AL-IdU/0/", "20170908-R9.5/BTF_AL_ONT_1_FAH14352_A-select_pass", 5]]

list_dir1 = [["20170908-R9.5/Human_AR", "20170908-R9.5/human_ar", 5]]
list_dir1 += [["20170908-R9.5/Human_HQ", "20170908-R9.5/human_hq", 5]]
# + list_dir[-3:]:  # + list_dir1:  # + list_dir[-1:]:
for dire, out, w in list_dir[1:4] + list_dir1:
    if redo:
        process(weights, directory="data/raw/%s/" % dire,
                output="data/processed/{0}{1}.fasta".format(basename, out), Nbases=args.nbases, reads="",
                filter=None, already_detected=False, Nmax=args.sample, size=args.size,
                n_output_network=1, n_input=1, chemistry="rf", window_size=w, clean=args.clean, old=False, res=args.residual,
                attention=args.attention)
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
