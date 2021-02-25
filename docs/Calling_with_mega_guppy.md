 `bin/guppy_basecaller -i in_path --save_path out_path --config data/modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg --fast5_out --device cuda:0`
 
Install megalodon
####
`conda create --name mega python=3.6 cython numpy pytorch=1.2`
`conda activate mega`
(IF guppy is the last version then)
pip install  megalodon

IF not:
then check guppy version:
`ont-guppy/bin/guppy_basecall_server -v`
if output:

Guppy Basecall Service Software, (C) Oxford Nanopore Technologies, Limited. Version 4.2.2+effbaf8, client-server API version 3.2.0
                                                                                      |
then install the corresponding version of pyguppy:

`pip install ont-pyguppy-client-lib==4.2.2 megalodon`


#Then copy  modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg and model_final_B.jsn in ont-guppy/data/

and create a symbolic link to ont-guppy in the repertory of execution


`megalodon in_path --outputs mod_mappings mod_basecall --reference S288C_reference_sequence_R64-2-1_20150113.fa  --processes 40    --overwrite --device cuda:0 --output-dir out_path  --guppy-config modif_from_dna_r9.4.1_450bps_modbases_dam-dcm-cpg_hac.cfg  --disable-mod-calibration`