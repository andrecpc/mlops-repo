mlops-repo
==============================

Final project by MLOps course (ODS, Yandex.Q)

ML task and a part of code are taken from here https://www.kaggle.com/code/chitwanmanchanda/vegetable-image-classification-using-cnn/notebook

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   └── raw            <- The original, immutable data dump.
    ├── Docker             <- Docker settings for minio, mlflow, pgsql, nginx
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    ├── mlruns             <- Meta-data of mlflows runnings
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks          <- Jupyter notebooks (EDA)
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── predict_sample.py
    │   │   └── train_model.py
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   |   └── visualize.py
    |   ├── app       <- Scripts to run API
    │       └── inference.py
    ├── venv               <- Virtual environment settings
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    └── docker-compose.yaml <- Docker settings
    └── dvc.lock           <- DVC meta-data
    └── dvc.yaml           <- DVC pipline settings 
    └── mlops-ods.drawio.xml    <- Pipline structure in draw.io format
    └── poetry path.txt    <- Usefull CLI commands 
    └── poetry.lock        <- Poetry meta-data 
    └── pyproject.toml     <- Poetry settings 
    └── start.py           <- Simple pythons pipline 
--------