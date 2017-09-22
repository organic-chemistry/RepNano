DeepNano5Bases
==============================

Extension of deepnano to work with 5 bases

To Evaluate the performances
==============================

To evaluate the model :

python -m src.models.predict_model --weights=data/training/my_model_weights-3390-removed-bad-B.h5 --directory=data/raw/control/ --Nbases=5 --output=data/processed/result.fasta

New model with extract_event:
python -m src.models.predict_model --weights=data/training/my_model_weights-9300.h5 --directory=data/raw/control/ --Nbases=5 --output=data/processed/result.fasta --detect

The directory must contain fasta sequences

to evaluate the model and get some info on T/B and alignement:
But the code must be modified

python -m src.test.evaluate

python -m src.models.train_model --Nbases 8 --pre-trained-weight=data/training/v9p5-delta10-ref-from-file-bis-max-files//my_model_weights-9300.h5 --from-pre-trained --pre-trained-dir-list=test-ref-all.txt --root data/training/v9p5-delta10-ref-from-net-bis-max-files-8b-max200 --deltaseq=50 --forcelength=0.1  --max-file=200

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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
