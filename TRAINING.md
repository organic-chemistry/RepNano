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