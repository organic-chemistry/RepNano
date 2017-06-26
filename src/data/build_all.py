import os
from .split_training import split
from .make_dataset import make


list_docs = [["temp", "sub_template.InDeepNano", "substituted"],
             ["comp", "sub_complement.InDeepNano", "substituted"],
             ["temp", "control_template.InDeepNano", "control"],
             ["comp", "control_complement.InDeepNano", "control"],
             ]
root = "data/raw/"
ref = "data/external/ref/S288C_reference_sequence_R64-2-1_20150113.fa"
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

    split(os.path.join(root, file_name), ref, output=os.path.join(processed, file_name))

    output_path = "{processed}/{folder}_{type_read}".format(folder=folder,
                                                            processed=processed, type_read=type_read)
    do_folder(output_path + "_train", exist_ok=True)

    make(type_read, processed + "/" + file_name + ".train",
         root + "/" + folder, output_path + "_train")
    do_folder(output_path + "_test", exist_ok=True)

    do_script("python src/data/make_dataset.py {type_read} {processed}/{file_name}.test {root}/{folder} {output_path}_test".format(root=root,
                                                                                                                                   file_name=file_name,
                                                                                                                                   ref=ref, folder=folder,
                                                                                                                                   output_path=output_path,
                                                                                                                                   type_read=type_read,
                                                                                                                                   processed=processed))
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
