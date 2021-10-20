From repnano directory (sym link for ont-guppy)
 python misc/preprocess_dataset.py --dataset ../RepNano/dataset_with_bug.csv --out /scratch/jarbona/preprocess_for_training_taiyaki_with_bug_from_model3 --from_scratch --cmd_mega '--guppy-config 3modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --outputs signal_mappings mod_mappings --disable-mod-calibration --device cuda:0 cuda:1 --process 20 --num-reads 500 ' --max_len 400

python misc/transform_signal_mapping.py --dataset dataset_transform_brdu_only_with_bug_subsample.csv --root_h5 /scratch/jarbona/preprocess_for_training_taiyaki_with_bug_from_model3//preprocess/ --alphabet ACGBT --collapsed_alphabet ACGTT --mod_long Brdu --output /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug_calibration_frommodl3.hdf5 --calibration

bin/train_flipflop.py --device 0 //scratch/jarbona/data_Repnano/alphabet_B_with_bug/model_checkpoint_00252.checkpoint  /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug_calibration_frommodl3.hdf5 --outdir /scratch/jarbona/data_Repnano/alphabet_B_Bug_balanced_dataset/ --save_every 200  --overwrite --lr_max 2.e-4

python misc/change_alphabet.py /scratch/jarbona/data_Repnano/alphabet_B_Bug_balanced_dataset/model_checkpoint_00007.checkpoint --output trained_B_balanced/ --alphabet ACGTB
python bin/dump_json.py trained_B_balanced/model_final.checkpoint --output trained_B_balanced/model_final.json

#TEsting:
 python misc/preprocess_dataset.py --dataset ../RepNano/dataset_with_bug.csv --out /scratch/jarbona/preprocess_for_training_taiyaki_with_bug_balance_testing --from_scratch --cmd_mega '--guppy-config 6modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --outputs signal_mappings mod_mappings --disable-mod-calibration --device cuda:0 cuda:1 --process 20 --num-reads 500 ' --max_len 400


 python misc/preprocess_dataset.py --dataset ../RepNano/dataset_with_bug.csv --out /scratch/jarbona/preprocess_for_training_taiyaki_with_bug_from_model3 --from_scratch --cmd_mega '--guppy-config 3modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --outputs signal_mappings mod_mappings --disable-mod-calibration --device cuda:0 cuda:1 --process 20 --num-reads 1200 ' --max_len 1000

 python misc/transform_signal_mapping.py --dataset dataset_transform_brdu_only_with_bug_subsample.csv --root_h5 /scratch/jarbona/preprocess_for_training_taiyaki_with_bug_from_model3/preprocess/ --alphabet ACGBT --collapsed_alphabet ACGTT --mod_long Brdu --output /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug_larger.hdf5 --calibration

 #checkc content
 python misc/plot_content_alphabet.py --h5 /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug_larger.hdf5 --mods B

bin/train_flipflop.py --device 0 models/mLstm_cat_mod_flipflop.py   /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug_larger.hdf5 --outdir /scratch/jarbona/data_Repnano/alphabet_B_Bug_balanced_dataset_from_scratch/ --save_every 250  --overwrite

python misc/change_alphabet.py /scratch/jarbona/data_Repnano//alphabet_B_Bug_balanced_dataset_from_scratch/model_checkpoint_00261.checkpoint --output trained_B_balanced/ --alphabet ACGTB


#From model4 (checkpoint 256)
 bin/train_flipflop.py  --device cuda:1 models/mLstm_cat_mod_flipflop.py /scratch/jarbona/data_Repnano/merged_all_Brdu_with_bug_4subs.hdf5 --outdir /scratch/jarbona/data_Repnano/alphabet_B_with_bug_subs_from4_st1/ --save_every 250 --stride 1 --chunk_len_min 1500 --chunk_len_max 3000  --min_sub_batch_size 32 --overwrite
