Training repnano on a sample with a given percent of incorporation
=====


You first need some samples with  known percent of brdu, and have sequenced them.
You need the fast5 files and the fastq files

You will then need to create two csv files. One to describe your dataset and another one
to describe the training.

The csv file to create the dataset must follow the structure here:

|key       |f5               |fq              |ref             |percents|mods |canonical|long_name|mix |
|----------|-----------------|----------------|----------------|--------|-----|---------|---------|----|
|Brdu_0.00 | /folder/01.fast5|/folder/01.fastq|/myfolder/ref.fa|[0]     |['B']|['T']    |['Brdu'] |True|
|Brdu_50.00| /folder/02.fast5|/folder/02.fastq|/myfolder/ref.fa|[50]    |['B']|['T']    |['Brdu'] |True|
|Brdu_80.00| /folder/03.fast5|/folder/03.fastq|/myfolder/ref.fa|[80]    |['B']|['T']   |['Brdu'] |True|


**key** must be a unique value tha will be used to reference a given sample during the training part
**f5** and **fq** must contain the absolute path to fast5 and fastq file. **ref** must contain the absolute path to the reference sequence.
percent is an array with the percent of modification of the sample. **mods** is the name of the base that will be used instead of the canonical base.
**mix** must specify is the training data is a mixture of 0 percent incorportation and a given percent so that the total percent of the sample
correspond to the value in the array percents. 


[//]: # (to manipulate dataset file `awk 'NR <=2 || NR>=12' dataset.csv > small_dataset.csv`)
Then you must execute the following code :

[//]: # (`python misc/preprocess_dataset.py --dataset dataset.csv --out out_folder --from_scratch --cmd_mega '--taiyaki-model-filename ../taiyaki/alphabet_B/model_final.checkpoint --process 8  --device 0 1 2 3  --outputs signal_mappings --do-not-use-guppy-server '`)
[//]: # (Possible to use guppy by creating a link to ont-guppy in megalodon and in RepNano)
[//]: # (For the calling to be runned from megalodon rep `python misc/preprocess_dataset.py --dataset dataset.csv --out out_folder --from_scratch --cmd_mega '--guppy-config modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --outputs signal_mappings mod_mappings --disable-mod-calibration --device cuda:0 --process 8'`) 
[//]: #(--disable-mod-calibration)

`python misc/preprocess_dataset.py --dataset dataset.csv --out out_folder --from_scratch --repnano_preprocess`

it will create two directories in the output folder:  initial_percent  preprocess and a csv file dataset_preprocessed.csv
This file is the copy of the previous file and also has a new column called **preprocessed** with the absolute path
to the h5 file which contains the alligned current base sequence.
The folder **preprocess** contains the preprocessed h5 files and **initial_percent** percents file that store the read file and the 
percent attributed to this read. At the beginning it will be the initial percent, but in case of a mixture the assign percent
will change with the training cycle.

`python src/repnano/models/training_loop.py  --dataset dataset_preprocessed.csv --training_info training.json`




Examples
`python misc/preprocess_dataset.py --dataset small_dataset_two.csv --out /scratch/jarbona/test_training_two/ --from_scratch --repnano_preprocess --max_len 100 --njobs 4`
`python src/repnano/models/training_loop.py  --dataset /scratch/jarbona/test_training_two/dataset_preprocessed.csv --training_info training_two_test.json`


# Trouble shoutting :
### Single to multi
if the read are single read one can use ont-fast5 api to create multi fast5 file eg:
`single_to_multi_fast5 --input_path input_path --save_path output --recursive --batch_size 4000`

### Unzipping part of a tar file:
list files into a tar archive
`tar -tvf BTF_AL_ONT_1_FAH14352_A_fast5.tgz`
extract a subdirectory
`tar -xf BTF_AP_ONT_1_FAH17611_A_fast5.tgz --wildcards --no-anchored 'sub/0/*'`

### when using guppy to align, fast_q numbers can be different from batch number 


python bin/change_alphabet.py ./alphabet_B_I/model_checkpoint_00003.checkpoint --output trained_B_I/
python bin/dump_json.py trained_B_I/model_final.checkpoint --output trained_B_I/model_final.json