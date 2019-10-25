Repnano
=============================

Wellcome to repnano the implementation of the software from ref.
Repnano allow to extract BrdU content from oxford nanopore experiment.

The software output two files:
on fasta file with the sequence where T have been replaced by T X or B according
to our transition matrix approach and a file fa_ratio_B with the corresponding
ratio for each base of the sequence as computed by the neural network

Install
==============================

### First install tombo :
```
conda create --name tomboenv python=3.6 keras pandas numba tqdm
conda activate tomboenv
conda install -c bioconda ont-tombo
```

Then modify _preprocess.py installed in tombo. To find the _preprocess file run :
```
conda config  --show envs_dirs
```

it should output the directory where is installed the python library: for example miniconda3/envs/

then _preprocess.py should be in miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/

You should the run:
```
cp modif_tombo/_preprocess.py miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/
```
### Then install repnano:
```
git clone https://github.com/organic-chemistry/RepNano.git
cd RepNano
python setup.py develop
```

Usage
=============================
The typical pipeline consist in the allignement of oxford nanopore reads and then
in the prediction of brdu content by the neural network


The typical output of nanopore is composed of 2 files:
one fastq file and one fast5 file.
If the fast5 file is compress the first step is to unzip it:
```
tar -xvzf data_fast5.tgz
```
This will create a directory 'unziped_dir' with a fast5 file inside

The first step is to separate the file that store 400 sequences in 400 one sequence files using the tool explode.py:
```
python src/repnano/data/explode.py unziped_dir/data.fast5 temporary_directory_to_store_the_400_files/
```
The next step is to associate the sequence of each read to the corresponding file using tombo :
```
tombo preprocess annotate_raw_with_fastqs --fast5-basedir temporary_directory_to_store_the_400_files/ --fastq-filenames data.fastq --overwrite --processes 4
```
If no file is process:
  - check if --overwrite is in the command line
  - check that you properly did the installation (replacing the preprocess.py file)
  - check that you have writing right on the files


Then we use tombo resquiggle command by alligning on the reference genome (here yeast: S288C_reference_sequence_R64-2-1_20150113.fa  )
```
tombo resquiggle temporary_directory_to_store_the_400_files/ S288C_reference_sequence_R64-2-1_20150113.fa --processes 4 --num-most-common-errors 5 --dna
```

Finally get the result with neural network and transition matrix
```
python src/repnano/models/predict_simple.py   --directory=temporary_directory_to_store_the_400_files/ --output=results/output_files.fa --overlap 10
```


