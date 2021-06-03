RepNano
=============================

Welcome to RepNano repository (implementation from [Hennion *et al.*, Genome Biol 21, 125. 2020](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02013-3)).
RepNano allows to estimate BrdU content from Oxford Nanopore raw sequencing reads.

The software outputs two files:
- a `.fa` fasta file with the read sequences where T have been replaced by T, X or B according
to our transition matrix (TM) approach
- a `.fa_ratio_B` file with the BrdU ratio for each base of the sequence as computed by the neural network (CNN). 


Installation
==============================

### First install Tombo :
```sh
conda create --name tomboenv --override-channels  -c defaults -c bioconda  python=3.6 keras pandas numba tqdm joblib  ont-tombo matplotlib
conda activate tomboenv
```


### Then install RepNano:
```sh
git clone https://github.com/organic-chemistry/RepNano.git
cd RepNano
python setup.py develop
```

Usage (on fast5 file that contains 4000 reads, 2019 and later)
==============================
The typical pipeline consists in Oxford Nanopore reads alignment on a reference genome followed by prediction of BrdU content 
by the neural network and the transition matrix approaches. 

Oxford Nanopore outputs 2 kinds of files:
- a fastq file containing all the sequences obtained by the Guppy basecaller,
- several fast5 files (raw currents) containing 4000 reads each.
If the fast5 folder is compressed, first run :
```
tar -xvzf data_fast5.tgz
```
This will create a directory with the .fast5 files inside. Then every .fast5 is processed separately (parallelizable).

The first step is to create a output.fast5 file that will contain the currents aligned on the reference genome (it uses
Tombo (ONT) package):
```sh
python src/repnano/data/preprocess.py  --hdf5 fast5_file.fast5 --fastq fastq_file.fastq --ref reference_genome.fa  --output_name output.fast5 --njobs 6
```
RepNano is then called on this file:
```sh
python src/repnano/models/predict_simple.py output.fast5 --bigf --output=BrdU_calls/output_file.fa --overlap 10
```
RepNano generates two files:
- **output_file.fa** with the read sequences where T have been replaced by T, X or B according
to our transition matrix (TM) approach
- **output_file.fa_ratio_B** with the BrdU ratio for each base of the sequence as computed by the neural network (CNN). 

Usage (Older files : one fast5 file per read, before 2019)
=============================

This pipeline requires additional installing steps to be done only once:

Additional installing steps
====

the \_preprocess.py file installed in tombo has to be modified. To find the _preprocess.py file to replace, run :
```sh
conda config  --show envs_dirs
```

It should output the directory where the python library is installed, for example miniconda3/envs/

In this example, the \_preprocess.py file to replace is in miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/

You should run:
```sh
cp modif_tombo/_preprocess.py miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/
```

Usage
====

If the data is compressed, first run :
```sh
tar -xvzf data_fast5.tgz
```
This will create one or several directories ('fast5_directory') with the fast5 files inside.

Then process every fast5 folder separately (parallelizable).

The first step is to associate the sequence of each read from the fastq file to the corresponding fast5 file using Tombo (see [Tombo documentation](https://github.com/nanoporetech/tombo)):
```sh
tombo preprocess annotate_raw_with_fastqs --fast5-basedir fast5_directory/ --fastq-filenames fastq_file.fastq --overwrite --processes 4

```

If the files are not processed:
  - check if --overwrite is in the command line
  - check that you properly did the installation (replacing the \_preprocess.py file)
  - check that you have write permission on the files


Then use Tombo `resquiggle` command to map the fastq sequence to the reference genome (here yeast: S288C_reference_sequence_R64-2-1_20150113.fa; alternatively a .mmi index generated by minimap2 can also be given) and to realign the raw currents to the reference sequence: 
```
tombo resquiggle fast5_directory/ S288C_reference_sequence_R64-2-1_20150113.fa --processes 4 --num-most-common-errors 5 --dna

```

Finally run RepNano to estimate BrdU content along mapped reads : 
```
python src/repnano/models/predict_simple.py   --directory=fast5_directory/ --output=BrdU_calls/output_files.fa --overlap 10
```

Fork detection, initiation and termination
==============================

The previous steps can be used to detect BrdU in any experimental context. In contrast, this last part is only to detect replication forks labelled in conditions resembling the ones used in [Hennion *et al.*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02013-3). 

Fork detection relies on the module [simplification](https://pypi.org/project/simplification/) that have to be installed first. 
```sh
pip install simplification
```

To detect replication forks, as well as initiation and termination events, you have to run the following command, where the 'BrdU_calls' folder is the output of RepNano, the 'DetectionFOLDER' is the location of the detection output files and 'prefix' is a prefix in the output files (it can be a sample ID for instance).

```sh
python src/repnano/detection/ForkPrediction-CNN-TM.py BrdU_calls/ DetectionFOLDER prefix
```
A number of parameters are set up at the beginning of ForkPrediction-CNN-TM.py and can be modified to make the detection more or less stringent.

The detection results obtained in [Hennion *et al.*](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02013-3) can be found in [Detected_events](Detected_events) folder (S288C yeast genome):
- [forks](Detected_events/FORKseq_TM-CNN.forks)
- [initiation events](Detected_events/FORKseq_TM-CNN.inits)
- [termination events](Detected_events/FORKseq_TM-CNN.term)

Additional scripts
==============================

Finally the folder *R* contains the R scripts used to perform the downstream analysis and to generate most of the figures of the article. 


License and acknowledgment
==============================

This software is licensed under the MIT license.  
During the course of development of this software, part of the software DeepNano
have been used (Boža, Vladimír, Broňa Brejová, and Tomáš Vinař. "DeepNano: deep recurrent neural networks for base calling in MinION nanopore reads." PloS one 12.6 (2017).)

TODO:
==============================

show error message when analysing R10 samples


create conda package / facilitate installation

separer resquiggle from mapping


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3743241.svg)](https://doi.org/10.5281/zenodo.3743241)

