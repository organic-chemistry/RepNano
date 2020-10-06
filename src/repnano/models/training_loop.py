import subprocess
import os
for i in range(8):
    if i == 0:
        input_folder = "data/preprocessed//training_initial/"
    else:
        input_folder = output_folder

    model_folder = f"training/from_initial_loop{i}/"
    output_folder = f"data/preprocessed//training_from_initial_network_{i}/"
    add=""
    if i >= 5:
        add = "--smalllr"

    if not os.path.exists(model_folder):
        cmd = f"python src/repnano/models/train_simple.py --root_data  {input_folder} --root_save {model_folder} --percents_training 0 17 28 35 55 59 73 80 --percents_validation 10 46 79 --error " + add
        #cmd = f"python src/repnano/models/train_simple.py --root_data  {input_folder} --root_save {model_folder} --percents_training 0  --percents_validation 0 --error"
        print(cmd)

        subprocess.run(cmd, shell=True, check=True)
    """
    try:
        cmd = f"python3 preprocess_dataset.py  --type dv --model {model_folder}/weights.hdf5 --output_dv {output_folder} --root_d /scratch/jarbona/data_Repnano/ --root_p data/preprocessed/ --ref /scratch/jarbona/repnanoV10/data/S288C_reference_sequence_R64-2-1_20150113.fa"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    except:
    """
    if not os.path.exists(output_folder):
        cmd = f"python3 preprocess_dataset.py  --type dv --model {model_folder}/weights.hdf5 --output_dv {output_folder} --root_d /scratch/jarbona/data_Repnano/ --root_p data/preprocessed/ --ref /scratch/jarbona/repnanoV10/data/S288C_reference_sequence_R64-2-1_20150113.fa"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

