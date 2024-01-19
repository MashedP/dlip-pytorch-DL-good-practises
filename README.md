# Dlip TP1 - Introduction to Pytorch and good practises
Welcome !

In this repository, you will find useful information  on good practises for Deep Learning research and developement projects.   

Note that those are general guidelines, recipes that evolves with the specificity of your project. The overall philosophy is "start small and incrementally add tools to understand what you are doing".

We prefer to use open-source solutions. 

This project will be in Python 3, using Pytorch. 

Have fun using this as a template and source of information for your first practical session & future projects !

The template by default contains code from the first exercise of Tp1. Try to adapt it for the second exercise.

*Inspired by [Solal Nathan's presentation](https://hebergement.universite-paris-saclay.fr/sepag/2023_05_24_Programming_Project_Management.pdf) on Programming project management*

# TODO

- [x] Packaging
- [x] Hydra
- [x] Mlflow
- [x] Pytorch Trainer
- [x] Pytorch Model
- [x] Evaluation 
- [x] Save and load Models from checkpoints 
- [ ] Study multiple models


# Project Organization



```
├── LICENSE            <- Information about the license of your code. Companies may have guidelines on how to license code. [See more here](https://choosealicense.com/)
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│ 
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt` Install the requirements by hand using `pip install -r requirements.txt`
│
├── pyproject.toml     <- Make this project pip installable with `pip install -e`
├── src/dlip           <- Source code for use in this project. The folder `dlip` is the package name you import
│   ├── __init__.py    <- Makes dlip a Python module
│   │
│   ├── conf           <- Store the configurations of your experiments (Hydra's yaml files )
│   │   └── train_linear.yaml
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py

```

This template is inspired by [Cookie Cutter](https://drivendata.github.io/cookiecutter-data-science/). Cookie Cutters  are templates of projects you can replicate and use as your own. They are great because their structure are familiar to other developper / ML engineer / Data Scientist. 

 

# Collaborative Work 

## Git Versioning

.gitkeep files are super useful to commit an empty folder
.gitignore are useful to exclude elements from the versioning ( heavy datasets for example) See templates of gitignore for your projects [here](https://github.com/github/gitignore)

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) Commits are traces of your changes on the project. Standards in commits leads to better communication between the actors of projects. Extensions exists for multiple projects 

[Git CheatSheet](https://training.github.com/downloads/github-git-cheat-sheet.pdf)

## Contributing

For contributions, have a look at Issues on gitlab for this project. 

You can fork this repo and suggest a merge request with information. Don't worry if you made a mistake/ introduced a bug, we are all here to learn ! 


# Tools usage 


## Seeds 

*seeds* are important to assess reproducibility of your results. Learn how they work in pytorch [here](https://pytorch.org/docs/stable/notes/randomness.html).


## Parsing Arguments

Common practise is to use a  parser in order to modify parameters you are using from the command line. In this repository, we use the Hydra package to handle arguments & configurations.


[Hydra](https://hydra.cc/docs/intro/) is a python package to handle parsing and configuration files 
	- based on [OmegaConf](https://github.com/omry/omegaconf)

Hydra allows you to parse arguments from the command line, to launch and log multiple experiments with different configuration easily. 

To run the training script, you can for example run 

```python src/dlip/models/train_model.py```

or with a different batch size :

```python src/dlip/models/train_model.py batch_size=20```

or launching multiple experiment on various batch sizes :

```python src/dlip/models/train_model.py --multirun batch_size=10,20,30,40,50```

It is very helpful when you have to perform sweeps on hyperparameters. 

Default Configurations and values of hyperparameters such as batch-size, learning rate, optimizers are stored in a yaml file ```src/dlip/conf/train_model.yaml```

You can access the configuration & outputs of previously run scripts by default in the ```outputs``` folder.

More info on how to setup hydra for your projects [here](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b)

## Logging experiment

MLflow, Tensorboard is the most common Experiment Tracking solution .  We chose to use [MLflow](https://www.mlflow.org/).

We integrate mlflow with hydra following this [blogpost](https://medium.com/optuna/easy-hyperparameter-management-with-hydra-mlflow-and-optuna-783730700e7d
)

Once you launched a training, you can visualize what is happening in your browser by launching ```mlflow ui``` in the terminal and opening the local link in your favorite browser Firefox 
Example : http://127.0.0.1:5000

## Code Packaging 

In python, the best way to load a module is to package it and install it.  There are several library for packaging code. Here we use [SetupTools](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)

`requirements.txt` is an important file for reproducibility. It contains all packages required to launch your experiment. 

After setting up your project, simply go to the root of the project and launch

```pip install -r requirements.txt  install -e . ```

You will be able to load the package inside the notebook. 

[More info on packaging](https://packaging.python.org/en/latest/)


## Python

General very useful guidelines for Python [here](https://docs.python-guide.org/) 


## Environment

It is recommended to build an environement for each of your different project. 
However, some modules like Pytorch are taking a lot of memory. 

 - [Virtual Env](https://realpython.com/python-virtual-environments-a-primer/)
 - [Conda](https://www.anaconda.com/download/)

 A more advanced way for reproducibility of your code is to contenerize your application, with tools like Docker. 

## Additional Guidelines & Informations 

For research code, paperwithcode guidelines are a good source of information:
Check them [here](https://github.com/paperswithcode/releasing-research-code/tree/master)