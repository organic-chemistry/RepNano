root="tests/input_bigfiles/"
ref="tests/ref_sequence/S288C_reference_sequence_R64-2-1_20150113.fa"
output_dir="tests/test_whole_pipeline_reference"
output_dir="tests/current_output"
python src/repnano/data/preprocess.py  --hdf5 $root/BTF_BY_ONT_1_FAK41381_A.NB11_0.fast5 --fastq $root/BTF_BY_ONT_1_FAK41381_A.NB11.fastq --ref ${ref}  --output_name ${output_dir}/output.fast5 --njobs 2 --max_len 40
python src/repnano/models/predict_simple.py ${output_dir}/output.fast5 --bigf --output=${output_dir}/output_file.fa --overlap 10

