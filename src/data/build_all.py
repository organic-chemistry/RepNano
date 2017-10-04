import os
from .split_training import split
from .make_dataset import make


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', dest='bwa', action='store_true')
parser.add_argument('--Nomake', dest='Nomake', action='store_false')
parser.add_argument('--root', dest='root', type=str)
parser.add_argument('--processed', dest='processed', type=str)
parser.add_argument('--file', dest='file', type=str)
parser.add_argument('--from-folder', dest='from_folder', type=str)
parser.add_argument('--type-read', dest="type_read", type=str,
                    default="temp", choices=["temp", "comp"])
parser.add_argument('--ref', dest='ref', type=str,
                    default="data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa")


args = parser.parse_args()


root = args.root
processed = args.processed
file_name = args.file
type_read = args.type_read
folder = args.from_folder
ref = args.ref


simulate = False
if simulate:
    do_script = print

    def do_folder(*args, **kwargs):
        print("create %s %s" % (str(args), str(kwargs)))
else:
    do_script = os.popen
    do_folder = os.makedirs


if not os.path.exists(ref) and args.bwa:
    print(ref, "not found")
    exit()
if not os.path.exists(os.path.join(root, file_name)):
    print(os.path.join(root, file_name), "not found")

if args.bwa:
    split(os.path.join(root, file_name), ref, output=os.path.join(processed, file_name))

if args.Nomake:
    output_path = "{processed}/{folder}_{type_read}".format(folder=folder,
                                                            processed=processed, type_read=type_read)

    do_folder(output_path + "_train", exist_ok=True)
    print("Reading", root + "/" + file_name + ".train")
    make(type_read, processed + "/" + file_name + ".train",
         root + "/" + folder, output_path + "_train")

    if args.bwa:
        do_folder(output_path + "_test", exist_ok=True)
        make(type_read, processed + "/" + file_name + ".test",
             root + "/" + folder, output_path + "_test")

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
