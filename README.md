RepNano
=============================

Wellcome to RepNano (implementation from ref).
RepNano allows to extract BrdU content from Oxford Nanopore raw reads.

The software outputs two files:
A fasta file with the sequence where T have been replaced by T, X or B according
to our transition matrix approach and a file .fa_ratio_B with the corresponding
BrdU ratio for each base of the sequence as computed by the neural network. 

Install
==============================

### First install Tombo :
```
conda create --name tomboenv python=3.6 keras pandas numba
conda activate tomboenv
conda install -c bioconda ont-tombo
```

Then the _preprocess.py file installed in tombo has to be modified. To find the _preprocess.py file to replace, run :
```
conda config  --show envs_dirs
```

It should output the directory where the python library is installed, for example miniconda3/envs/

In this example, the _preprocess.py file to replace is in miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/

You should run:
```
cp modif_tombo/_preprocess.py miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/
```
### Then install RepNano:
```
git clone https://github.com/organic-chemistry/RepNano.git
cd RepNano
python setup.py develop
```

Usage
=============================
The typical pipeline consists in Oxford Nanopore reads alignment on a reference genome followed by prediction of BrdU content by the neural network and the transition matrix approaches. 

Oxford Nanopore outputs 2 kind of files:
A fastq file containing all the sequences obtained by the Guppy basecaller, and several fast5 files (raw currents) containing 4000 reads each.

If the fast5 folder is compressed, first run :
```
tar -xvzf data_fast5.tgz
```
This will create a directory 'unziped_dir' with the fast5 files inside.

Then process every fast5 file separately (parallelizable).

As Tombo process only one-read containing fast5, the first step is to separate the file that store 4000 sequences in 4000 one-sequence files using the tool explode.py:
```
python src/repnano/data/explode.py unziped_dir/data.fast5 temporary_directory_to_store_the_4000_files/
```
The next step is to associate the sequence of each read from the fastq file to the corresponding fast5 file using Tombo (see Tombo documentation):
```
tombo preprocess annotate_raw_with_fastqs --fast5-basedir temporary_directory_to_store_the_4000_files/ --fastq-filenames data.fastq --overwrite --processes 4
```

If no file is processed:
  - check if --overwrite is in the command line
  - check that you properly did the installation (replacing the _preprocess.py file)
  - check that you have write permission on the files


Then use Tombo resquiggle command to map the fastq sequence on the reference genome (here yeast: S288C_reference_sequence_R64-2-1_20150113.fa) and to realign the raw current on the reference sequence: 
```
tombo resquiggle temporary_directory_to_store_the_4000_files/ S288C_reference_sequence_R64-2-1_20150113.fa --processes 4 --num-most-common-errors 5 --dna
```

Finally run RepNano to estimate BrdU content along mapped reads : 
```
python src/repnano/models/predict_simple.py   --directory=temporary_directory_to_store_the_4000_files/ --output=results/output_files.fa --overlap 10
```


