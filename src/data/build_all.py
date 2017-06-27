import os
from .split_training import split
from .make_dataset import make


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', dest='bwa', action='store_true')
parser.add_argument('--Nomake', dest='Nomake', action='store_false')

args = parser.parse_args()


list_docs = [["temp", "sub_template.InDeepNano", "substituted"],
             ["comp", "sub_complement.InDeepNano", "substituted"],
             ["temp", "control_template.InDeepNano", "control"],
             ["comp", "control_complement.InDeepNano", "control"],
             ]
root = "data/raw/"
ref = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"
processed = "data/processed/"


list_docs = [["temp", "controlK47211_template.InDeepNano", "control-K47211"],
             ["comp", "controlK47211_complement.InDeepNano", "control-K47211"]]
root = "data/raw/"
#ref = "data/external/K47211/controlK47211.fa"
processed = "data/processed/"


simulate = False
if simulate:
    do_script = print

    def do_folder(*args, **kwargs):
        print("create %s %s" % (str(args), str(kwargs)))
else:
    do_script = os.popen
    do_folder = os.makedirs

for type_read, file_name, folder in list_docs:

    if not os.path.exists(ref):
        print(ref, "not found")
    if not os.path.exists(os.path.join(root, file_name)):
        print(os.path.join(root, file_name), "not found")

    if args.bwa:
        split(os.path.join(root, file_name), ref, output=os.path.join(processed, file_name))

    if args.Nomake:
        output_path = "{processed}/{folder}_{type_read}".format(folder=folder,
                                                                processed=processed, type_read=type_read)
        do_folder(output_path + "_train", exist_ok=True)

        make(type_read, processed + "/" + file_name + ".train",
             root + "/" + folder, output_path + "_train")
        do_folder(output_path + "_test", exist_ok=True)

        make(type_read, processed + "/" + file_name + ".test",
             root + "/" + folder, output_path + "_test")
        do_folder(output_path + "_test", exist_ok=True)

  #  python split_training.py $root/$element $external/S288C_reference_sequence_R64-2-1_20150113.fa
    # python prepare_dataset.py temp $root/$element.train ../ForJM/control/
    # ../ForJM/control_template_train


#
# python split_training.py  ../ForJM/sub_template.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
# python split_training.py  ../ForJM/control_template.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
#
# python prepare_dataset.py temp ../ForJM/control_template.InDeepNano.train ../ForJM/control/ ../ForJM/control_template_train
# python prepare_dataset.py temp ../ForJM/control_template.InDeepNano.test ../ForJM/control/ ../ForJM/control_template_test
# python prepare_dataset.py temp ../ForJM/sub_template.InDeepNano.test ../ForJM/substituted/ ../ForJM/sub_template_test
# python prepare_dataset.py temp ../ForJM/sub_template.InDeepNano.train ../ForJM/substituted/ ../ForJM/sub_template_train
#
#
# python split_training.py  ../ForJM/control_complement.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
# python split_training.py  ../ForJM/sub_complement.InDeepNano   ../ref/S288C_reference_sequence_R64-2-1_20150113.fa
#
#
# mkdir ../ForJM/sub_complement_train
# mkdir ../ForJM/sub_complement_test
# mkdir ../ForJM/control_complement_test
# mkdir ../ForJM/control_complement_train
#
# python prepare_dataset.py comp ../ForJM/sub_complement.InDeepNano.train ../ForJM/substituted/ ../ForJM/sub_complement_train
# python prepare_dataset.py comp ../ForJM/sub_complement.InDeepNano.test ../ForJM/substituted/ ../ForJM/sub_complement_test
# python prepare_dataset.py comp ../ForJM/control_complement.InDeepNano.test ../ForJM/control/ ../ForJM/control_complement_test
# python prepare_dataset.py comp
# ../ForJM/control_complement.InDeepNano.train ../ForJM/control/
# ../ForJM/control_complement_train
