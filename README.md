Repnano
=============================

Wellcome to repnano the implementation of the software from ref.
Repnano allow to extract BrdU content from oxford nanopore experiment.

The software output two files:
one fasta file with the sequence where T have been replaced by T X or B according
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


### Then install repnano:
```
git clone https://github.com/organic-chemistry/RepNano.git
cd RepNano
python setup.py develop
```

Usage (on fast5 file that contains 4000 reads)
==============================
The first step is to create a file: output.fast5 that will contain the aligned current on the reference genome by using
tombo package:
```
python src/repnano/data/preprocess.py  --hdf5 fast5_file.fast5 --fastq fastq_file.fastq --ref reference_genome.fa  --output_name output.fast5 --njobs 6
```
Then to call repnano on this file:
```
python src/repnano/models/predict_simple.py output.fast5 --bigf   --output=results/output_file.fa --overlap 10
```
repnano generate two files:
one fasta file (output_file.fa) with the sequence where T have been replaced by T X or B according
to our transition matrix approach and a file (output_file.fa_ratio_B) with the corresponding
ratio for each base of the sequence as computed by the neural network

Usage (Old) on fast5 file that contain one read
=============================

The typical pipeline consist in the allignement of oxford nanopore reads and then
in the prediction of brdu content by the neural network


The typical output of nanopore is composed of 2 files:
one fastq file and one fast5 file.
If the fast5 file is compress the first step is to unzip it:


The first step is to associate the sequence of each read to the corresponding file using tombo :
```
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5_directory/ --fastq-filenames data.fastq --overwrite --processes 4
```
If no file is process:
  - check if --overwrite is in the command line
  - check that you properly did the installation (replacing the preprocess.py file)
  - check that you have writing right on the files


Then we use tombo resquiggle command by alligning on the reference genome
```
tombo resquiggle fast5_directory/ reference_genome.fa --processes 4 --num-most-common-errors 5 --dna
```

Finally get the result with neural network and transition matrix
```
python src/repnano/models/predict_simple.py   --directory=temporary_directory_to_store_the_400_files/ --output=results/output_files.fa --overlap 10
```


