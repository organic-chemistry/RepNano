from ..models.predict_model import process
import os
import subprocess


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

S = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1))
KTF.set_session(S)

weights = "data/cluster/training/v9p5delta10-bis/my_model_weights-940.h5"
# weights = "data/cluster/training/v9p5delta10-new-weight-longer/my_model_weights-60.h5"  # a lot of B
# weights = "data/cluster/training//v9p5-delta10-oversamplingL/my_model_weights-190.h5"
# weights = "data/cluster/training//v9p5-delta10-ref-from-file-only-T/my_model_weights-470.h5"
# weights = "data/cluster/training//v9p5-delta10-ref-from-file/my_model_weights-50.h5"

# weights = "data/cluster/training/v9p5-delta10-oversamplingB/my_model_weights-20.h5"
weights = "data/cluster/training/v9p5-delta10-ref-from-file-bis-max-files/my_model_weights-9300.h5"
weights = "data/training/al-8-bases/my_model_weights-90.h5"
weights = "data/training/test-single-base/my_model_weights-180.h5"
weights = "data/cluster/training/test-single-base-filter/my_model_weights-60.h5"
weights = "data/training/my_model_weights-3390-removed-bad-B.h5"
# weights = "data/cluster/training/skip-new/my_model_weights-7590.h5"
# weights = "data/cluster/training/allign-agree-five/my_model_weights-2560.h5"
weights = "data/cluster/training/allign-agree-five-clean-B/my_model_weights-3990.h5"
# weights = "data/cluster/training/test-single-various-w-size-8bases/my_model_weights-30.h5"
# weights = "data/training/test-single-base-bis/my_model_weights-0.h5"
weights = "data/cluster/training/test-single-various-w-size-8bases-smallerw/my_model_weights-1800.h5"
weights = "data/cluster/training/allign-agree-five-clean-B-smallB/my_model_weights-5990.h5"
weights = "data/cluster/training/allign-agree-five-clean-B-smallB-test-ssample/my_model_weights-2480.h5"
weights = "data/cluster/training/test-single-various-w-size-8bases-smallw-fixed/my_model_weights-500.h5"


basename = "results/v9p5-best-B-20170908-R9.5-newchem-test-clean-window_size/"

weights = "data/cluster/training/allign-agree-85555/my_model_weights-3900.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-bw/"

"""
weights = "data/cluster/training/allign-agree-63333/my_model_weights-1160.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-sw/"




"""
weights = "data/training/my_model_weights-3390-removed-bad-B.h5"
basename = "results/ref/"

weights = "data/cluster/training/allign-agree-85555-ctc20/my_model_weights-3990.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-bw-ctc20/"

"""
weights = "data/cluster/training/allign-agree-85555-8b/my_model_weights-1960.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-8test/"

"""
# weights = "data/cluster/training/allign-agree-85555/my_model_weights-3900.h5"
# basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-bw/"
"""
weights = "data/cluster/training/allign-agree-85555-BI-ctc20/my_model_weights-1990.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-bw-ctc20-BI/"


weights = "data/cluster/training/allign-agree-63333-8b/my_model_weights-1950.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-8test/"
"""


weights = "data/cluster/training/allign-agree-85555-ctc200/my_model_weights-280.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-bw-ctc200/"

"""
weights = "data/cluster/training//clean_scale_85555-ctc50-8B/my_model_weights-1990.h5"
basename = "results/clean-ctc50-8B/"

weights = "data/cluster/training//clean_scale_l3_85555-ctc50/my_model_weights-790.h5"
basename = "results/clean-l3-ctc50/"

weights = "data/cluster/training/ref_85555-ctc50-drop/my_model_weights-2670.h5"
basename = "results/ref/"

weights = "data/cluster/training/clean_two_scale_l3_85555-ctc50-drop-clean-B-lr0p001/my_model_weights-810.h5"
basename = "results/clean-l3-clean/"

weights = "data/cluster/training/clean_two_scale_l3_85555-ctc200-agree-align-cleanB/my_model_weights-160.h5"
basename = "results/clean-l3-noise/"

"""

weights = "data/cluster/training/agree-align-cleanB-ctc200//my_model_weights-110.h5"
basename = "results/clean-ctc200-ramp/"


weights = "data/cluster/training/allign-no-agree-85555-ctc200/my_model_weights-190.h5"
basename = "results/no-agree-ctc200-no-agree/"
"""
weights = "data/cluster/training/clean_two_scale_l3_85555-ctc200-agree-align-cleanB/my_model_weights-160.h5"
basename = "results/clean-l3-noise/"

weights = "data/cluster/training/allign-no-agree-85555-ctc200/my_model_weights-300.h5"
basename = "results/no-agree-ctc200-no-agree-w300/"


weights = "data/cluster/training/allign-agree-85555/my_model_weights-3900.h5"
basename = "results/v9p5-best-B-20170908-R9.5-froms-two-200-seg-last-smaller-explo-bw/"


weights = "data/cluster/training/allign-agree-85555-ctc200-residual/my_model_weights-1060.h5"
basename = "results/resid/"

weights = "data/cluster/training//allign-no-agree-85555-ctc200-from-improoved/my_model_weights-390.h5"
basename = "results/from-improved/"
"""

weights = "data/cluster/training/allign-agree-85555-ctc20-attention/my_model_weights-1990.h5"
basename = "results/attention/"


weights = "data/cluster/training/allign-agree-85555-ctc200-residual-attention/my_model_weights-460.h5"
basename = "results/res-attention/"

weights = "data/cluster/training/allign-agree-85555-ctc200-other-bases/my_model_weights-1070.h5"
basename = "results/other-base/"

weights = "data/cluster/training/allign-agree-85555-ctc200-clean-test/my_model_weights-10.h5"
basename = "results/clean-test/"


# weights = "data/training/my_model_weights-3390-removed-bad-B.h5"
# basename = "results/ref-nodetect/"

weights = "data/cluster/training/allign-no-agree-85555-ctc200/my_model_weights-190.h5"
basename = "results/no-agree-ctc200-no-agree/"

weights = "data/cluster/training/training_set_from_old_pre_trained/my_model_weights-400.h5"
basename = "fresh_pre_trained"
"""
weights = "data/cluster/training/allign-agree-85555-ctc400/my_model_weights-340.h5"
basename = "results/cctc400/"
"""

weights = "data/cluster/training/training_set_from_old_residulal_clean/my_model_weights-530.h5"
basename = "fresh_pre_trained-clean"


weights = "data/cluster/training/training_set_from_old_residulal_clean/my_model_weights-530.h5"
basename = "fresh_pre_trained-clean-mw"

weights = "data/cluster/training/training_set_from_old_residulal_clean/my_model_weights-530.h5"
basename = "fresh_pre_trained-clean-mw-median"

weights = "data/cluster/training/training_set_from_old_residual_clean-correct-std/my_model_weights-490.h5"
basename = "results/clean-std-ctc200/"

"""
weights = "data/cluster/training/training_set_from_old_residual_attention_clean-correct-std-ctc50/my_model_weights-440.h5"
basename = "results/clean-std-ctc200-attention/"


weights = "data/training/my_model_weights-3390-removed-bad-B.h5"
basename = "results/ref/"

weights = "data/cluster/training/allign-no-agree-85555-ctc200/my_model_weights-190.h5"
basename = "results/no-agree-ctc200-no-agree-clean-std/"


weights = "data/cluster/training//training_set_from_best_so_far_residual_clean_ctc200/my_model_weights-60.h5"
basename = "results/from_best_resi_cleanctc200/"

# weights = "data/cluster/training/training_set_from_best_so_far_residual_clean_ctc20/my_model_weights-280.h5"
# basename = "results/from_best_resi_cleanctc20/"

# weights = "data/cluster/training/training_set_from_best_so_far_residual_attention_clean_ctc20/my_model_weights-280.h5"
# basename = "results/from_best_resi_att_cleanctc20/"
"""

weights = "data/training/my_model_weights-3390-removed-bad-B.h5"
basename = "results/ref-std-clean/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200/my_model_weights-490.h5"
basename = "results/std-clean/"

# weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc20/my_model_weights-330.h5"
# basename = "results/std-clean-ctc20/"

# weights = "data/cluster/training/training_correct-std_0p1_residual_attention_clean_ctc20/my_model_weights-300.h5"
# basename = "results/std-clean-att-ctc20/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-b8/my_model_weights-540.h5"
basename = "results/std-clean-b8/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta200/my_model_weights-490.h5"
basename = "results/std-clean-delta200/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-b8-realign/my_model_weights-290.h5"
basename = "results/std-clean-b8/"

# weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-realign/my_model_weights-290.h5"
# basename = "results/std-clean-delta400/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-realign/my_model_weights-290.h5"
basename = "results/std-clean-bTBI/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-clean-I-realign/my_model_weights-580.h5"
basename = "results/std-clean-bTBI-cleanI/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBE-clean-E-realign/my_model_weights-500.h5"
basename = "results/std-clean-bTBE-cleanE/"
# weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-b8-realign/my_model_weights-290.h5"
# basename = "results/std-clean-b8/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-clean-I-realign-longer/my_model_weights-450.h5"
basename = "results/std-clean-bTBI-cleanI-longer/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-realign/my_model_weights-290.h5"
basename = "results/std-clean-delta400/"

weights = "data/cluster/training/training_correct-std_0p1_residual_clean_ctc200-delta400-realign/my_model_weights-290.h5"
basename = "results/std-clean-delta400-test/"


weights = "data/cluster/training_correct-std_0p1_residual_clean_ctc200-delta400-all-T/my_model_weights-480.h5"
basename = "results/std-clean-delta400-test-allT/"

weights = "data/training/training_correct-std_0p1_residual_clean_ctc200-delta400-realign_my_model_weights-290.h5"
basename = "results/test"

weights = "data/training/my_model_weights-3390-removed-bad-B.h5"
basename = "results/test/"

weights = "../../..//tmp/test/my_model_weights-70.h5"
basename = "results/test-2/"

weights = "data/training/hybrid-ctc200.h5"
basename = "results/test-hybrid-thres20/"

weights = "data/cluster//hybrid-TBI/my_model_weights-0.h5"
#weights = "data/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-clean-I-realign_my_model_weights-580.h5"
basename = "results/hybrid-TBI/"
#weights = "data/training/my_model_weights-3390-removed                                                                                                                                                                                                 -bad-B.h5"
#basename = "results/ref-std-clean/"

weights = "data/cluster//hybrid-TBI-small-lr/my_model_weights-0.h5"
#weights = "data/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-clean-I-realign_my_model_weights-580.h5"
basename = "results/hybrid-TBI-small-lr/"

weights = "data/cluster//hybrid-TBI-smaller-lr/my_model_weights-0.h5"
#weights = "data/training/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-clean-I-realign_my_model_weights-580.h5"
basename = "results/hybrid-TBI-smaller-lr/"


weights = "data/training/hybrid-ctc200.h5"
basename = "results/test-hybrid/"


weights = "data/training/hybrid-ctc200.h5"
basename = "results/test-hybrid/"

"""
weights = "data/cluster/training_correct-std_0p1_residual_clean_ctc200-delta400-bTBI-clean-I-realign/my_model_weights-580.h5"
basename = "results/std-clean-bTBI-cleanI/"
"""

weights = "data/training/training_correct-std_0p1_residual_clean_ctc200-delta400-realign_my_model_weights-290.h5"
basename = "results/test1"

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
default = list_dir[1:4] + list_dir[-3:-2] + list_dir1
default = list_dir[3:4]  # + list_dir1
for dire, out, w in default:
    if redo:
        process(weights, directory="data/raw/%s/" % dire,
                output="data/processed/{0}{1}.fasta".format(basename, out), Nbases=5, reads="",
                filter=None, already_detected=False, Nmax=10, size=40,
                n_output_network=1, n_input=1, chemistry="rf", window_size=w, clean=True, old=False, res=True,
                attention=False)
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
