Training the different steps:
=========

Download fastq and fast5 at different %
# download_all.sh
# Decompress  cat list_files.txt | xargs -n 1 -I {} dtrx {}
# cat list_files.txt | awk -F"[.]" '{printf"%s.%s_fast5\n",$1,$2}'  | xargs -n 1 -I {}  mkdir {}
# cat list_files.txt | awk -F"[.]" '{printf"%s.%s_fast5\n",$1,$2}'  | xargs -n 1 -I {}  tar -zxvf  {}.tgz -C {}

# scrip preprocess_dataset

`python3 preprocess_dataset.py --root_d test_set_brdu --root_p data/preprocessed --from_scratch`



Preprocess according to Readme, this create the preprocessed file named by percent.
and create a list of repertory files that contain direction to preprocess read and their initial percent
Then the calling can be made with the previous best model.






	% B 	% masse spec
BO 	0 	0
BP 	10 	9,4
BQ 	20 	16,6
BR 	30 	27,9
BS 	40 	35,1
BT 	50 	46,1
BU 	60 	54,8
BV 	70 	59,0
BW 	80 	72,6
BX 	90 	78,8
BY 	100  80,3


with gupyy
#######################

to put all h5 at the same place:
`python3 preprocess_dataset_guppy.py   --root_d /scratch/jarbona/data_Repnano/ --root_p /scratch/jarbona/data_Repnano/grouped_list/ --ref /scratch/jarbona/repnanoV10/data/S288C_reference_sequence_R64-2-1_20150113.fa`


to create scaling file
`bin/generate_per_read_params.py /scratch/jarbona/data_Repnano/grouped_list/ --outpu /scratch/jarbona/data_Repnano/scaling/scale.tsv`


to merge fa
` ls ~/RepNano/data/preprocessed/test_with_error_corrected_17_test_lstm//*.fa | xargs -I {} cat {} > /home/jarbona/RepNano/data/preprocessed/test_with_error_corrected_17_test_lstm/merged.fa`

to get read of /read
`cat /home/jarbona/RepNano/data/preprocessed/test_with_error_corrected_17_test_lstm/merged.fa |  sed 's|/read_||g' > /home/jarbona/RepNano/data/preprocessed/test_with_error_corrected_17_test_lstm/merged_trimmed.fa`


to prepare reads for training (taiyaki):
`bin/prepare_mapped_reads.py /scratch/jarbona/data_Repnano/grouped_list/  /scratch/jarbona/data_Repnano/scaling/scale.tsv /scratch/jarbona/data_Repnano/mapped_signal_file.hdf5  models/mGru_flipflop_remapping_model_r9_DNA.checkpoint /home/jarbona/RepNano/data/preprocessed/test_with_error_corrected_17_test_lstm/merged_trimmed.fa --mod B T Brdu --job 6`

It can also be done on megalodon part


to train

`bin/train_flipflop.py --device 0 models/mLstm_cat_mod_flipflop.py /scratch/jarbona/data_Repnano/mapped_signal_file_mod_base.hdf5 --outdir alphabet_B_smaller/ --save_every 250`

Then must correct for alphabet (taiyaki/bin/change_alphabet.py)


to map:

`megalodon /scratch/jarbona/data_Repnano/human/  --outputs mod_mappings --output-directory /scratch/jarbona/data_Repnano/human_basecall/ --do-not-use-guppy-server --taiyaki-model-filename ../taiyaki/alphabet_B/model_final.checkpoint --process 4 --overwrite --device 0   --reference /scratch/jarbona/ref/human/all_chra.fa  --disable-mod-calibration`

to bascall

mod_basecalls

Merged env:
conda create -c bioconda --name merged python=3.6 cython numpy pytorch=1.2  keras pandas numba tqdm joblib  ont-tombo matplotlib

#install megalodon and taiyaki

New training
#################
#create a dataset file (detail)
`python preprocess_dataset.py --dataset dataset.csv --out out_folder --from_scratch --cmd_mega '--taiyaki-model-filename ../taiyaki/alphabet_B/model_final.checkpoint --process 8  --device 0 1 2 3'`
this create a local file called dataset_preprocessed.csv 

Then create a file called training.json:
python src/repnano/models/training_loop.py  --dataset dataset_preprocessed.csv --training_info training.json


Calling with gupy classic
#################
 `bin/guppy_basecaller /scratch/jarbona/data_Repnano/grouped_list_small_val/ --save_path /scratch/jarbona/data_Repnano/grouped_list_small_val_bassecalled --config data/dna_r9.4.1_450bps_hac.cfg --device cuda:0`