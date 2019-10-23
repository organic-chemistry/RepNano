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
conda create --name tomboenv python=3.6 keras pandas numba
conda activate tomboenv
conda install -c bioconda ont-tombo

then modify _preprocess.py installed in tombo
to find the _preprocess file run :
conda config  --show envs_dirs
s

it should output the directory where is installed the python library: for example
miniconda3/envs/
then preprocess should be in
miniconda3/envs/tomboenv/lib/python3.6/site-packages/tombo/

(typically in directory similar to miniconda3new/envs/keras3/lib/python3.6/site-packages/tombo/)
by the version in modif_tombo/_preprocess.py

### Then install repnano:
git clone https://github.com/organic-chemistry/RepNano.git
cd RepNano
python setup.py develop

Usage
=============================
The typical pipeline consist in the allignement of oxford nanopore reads and then
in the prediction of brdu content by the neural network


The typical output of nanopore is composed of 2 files:
one fastq file and one fast5 file.
If the fast5 file is compress the first step is to unzip it:
tar -xvzf data_fast5.tgz
This will create a directory 'unziped_dir' with a fast5 file inside

The first step is to separate the file that store 400 sequences in 400 one sequence files using the tool explode.py:

python src/repnano/data/explode.py unziped_dir/data.fast5 temporary_directory_to_store_the_400_files/

The next step is to associate the sequence of each read to the corresponding file using tombo :

tombo preprocess annotate_raw_with_fastqs --fast5-basedir temporary_directory_to_store_the_400_files/ --fastq-filenames data.fastq --overwrite --processes 4

If no file is process:
  - check if --overwrite is in the command line
  - check that you properly did the installation (replacing the preprocess.py file)
  - check that you have writing right on the files


Then we use tombo resquiggle command by alligning on the reference genome (here yeast: S288C_reference_sequence_R64-2-1_20150113.fa  )

tombo resquiggle temporary_directory_to_store_the_400_files/ S288C_reference_sequence_R64-2-1_20150113.fa --processes 4 --num-most-common-errors 5 --dna


python src/repnano/models/predict_simple.py   --directory=temporary_directory_to_store_the_400_files/ --output=results/output_files.fa --overlap 10





To predict with the model
==============================
==============================

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├──build_all.py        <- Given InDeepNano files, fasta file corresponding to the indeepnano and ref of the genome
    │   │   │                         generate automatically training and test set  
    │   │   ├──split_training.py   <- Given InDeepNano files and ref of the genome generate a train and test InDeepNano file
    │   │   │                         where test contain only the chromosome 11
    │   │   └── make_dataset.py    <- Given one Basecall files and a fasta files generate the data to train the model
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── helpers.py         <- tools to normalize the signal
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model.py           <- Generate keras model
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── test           <- Scripts to evaluate the results on the test set
    │   │   ├──evaluate.py                    <- Given InDeepNano files, fasta file corresponding to the indeepnano and ref of the genome
    │   │   ├──ExportStatAlnFromSamYeast.py    <- script from magalie te get stat on alignements
    │   │   └──get_fasta_from_train-test.py    <- script that get the results from the model and give stats on T and B and generate fasta without B
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

