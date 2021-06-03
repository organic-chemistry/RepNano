Creating mappping dataset

`python misc/preprocess_dataset.py --dataset ../RepNano/dataset_fork_idu.csv --out /scratch/jarbona/preprocess_for_training_taiyaki --from_scratch --cmd_mega '--guppy-config modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --outputs signal_mappings mod_mappings --disable-mod-calibration --device cuda:0 --process 20 '`

Merging and renamming

`python misc/transform_signal_mapping.py --dataset dataset_transform.csv --root_h5 /scratch/jarbona/preprocess_for_training_taiyaki/preprocess/ --alphabet ACGBIT --collapsed_alphabet ACGTTT --mod_long Brdu Idu --output /scratch/jarbona/data_Repnano/merged_all_Brdu_Idu.hdf5`

Check content

`python misc/plot_content_alphabet.py --h5 /scratch/jarbona/data_Repnano/small.hdf5`


`bin/train_flipflop.py --device 0 models/mLstm_cat_mod_flipflop.py /scratch/jarbona/data_Repnano/mapped_signal_file_mod_base.hdf5 --outdir alphabet_B_smaller/ --save_every 250`


python bin/change_alphabet.py ./alphabet_B_I/model_checkpoint_00003.checkpoint --output trained_B_I/
python bin/dump_json.py trained_B_I/model_final.checkpoint --output trained_B_I/model_final.json