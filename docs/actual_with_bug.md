From repnano directory (sym link for ont-guppy)
python misc/preprocess_dataset.py --dataset ../RepNano/dataset_with_bug.csv --out /scratch/jarbona/preprocess_for_training_taiyaki_with_bug --from_scratch --cmd_mega '--guppy-config modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --outputs signal_mappings mod_mappings --disable-mod-calibration --device cuda:0 --process 20 ' --max_len 400

python misc/transform_signal_mapping.py --dataset dataset_transform_brdu_only_with_bug.csv --root_h5 /scratch/jarbona/preprocess_for_training_taiyaki_with_bug/preprocess/ --alphabet ACGBT --collapsed_alphabet ACGTT --mod_long Brdu --output /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug.hdf5

bin/train_flipflop.py --device 0 models/mLstm_cat_mod_flipflop.py /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug.hdf5 --ou
tdir /scratch/jarbona/data_Repnano/alphabet_B_with_bug/ --save_every 250