root="tests/input_bigfiles/"
ref="tests/ref_sequence/S288C_reference_sequence_R64-2-1_20150113.fa"
ref_dir="tests/test_whole_pipeline_reference"
output_dir="tests/current_output_calling"
python src/repnano/models/predict_simple.py ${ref_dir}/output.fast5 --bigf --output=${output_dir}/output_file.fa --overlap 10

