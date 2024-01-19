# Dlip TP1 - Introduction to Pytorch and good practises
Welcome !

In this repository, you will find useful information  on good practises for Deep Learning research and developement projects.   

Note that those are general guidelines, recipes that evolves with the specificity of your project. We tend to prefer open-source solutions. 

This project will be in Python 3, using Pytorch. 

Have fun using this as a template and source of information for your first practical session & future projects !

*Inspired by [Solal Nathan's presentation](https://hebergement.universite-paris-saclay.fr/sepag/2023_05_24_Programming_Project_Management.pdf) on Programming project management*

# TODO
- [x] Packaging
- [x] Hydra
- [ ] Mlflow
- [x] Pytorch Trainer
- [x] Pytorch Model
- [ ] Evaluation 
- [x] Save and load Models from checkpoints 


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

The template by default contains code from the first exercise of Tp1. 

# Collaborative Work 

## Git Versioning

.gitkeep files are super useful to commit an empty folder
.gitignore are useful to exclude elements from the versioning ( heavy datasets for example) See templates of gitignore for your projects [here](https://github.com/github/gitignore)

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) Commits are traces of your changes on the project. Standards in commits leads to better communication between the actors of projects. Extensions exists for multiple projects 

[Git CheatSheet](https://training.github.com/downloads/github-git-cheat-sheet.pdf)


## Configuration


### Seeds 

*seeds* are important to assess reproducibility of your results. Learn how they work in pytorch [here](https://pytorch.org/docs/stable/notes/randomness.html).

`requirements.txt` is an important file for reproducibility. It contains all packages required to launch your experiment. 

### Parsing Arguments


Common practise is to use an argparse parser. In this repository, we use the Hydra package to handle arguments & configurations.


[Hydra](https://hydra.cc/docs/intro/) is a python package to handle parsing and configuration files 
	- based on [OmegaConf](https://github.com/omry/omegaconf)

Hydra allows you to parse arguments from the command line, to launch and log multiple experiments with different configuration easily. 

To run the training script, you can for example run 

```python src/dlip/models/train_model.py```

or with a different batch size :

```python src/dlip/models/train_model.py batch_size=20```

It is very helpful when you have to perform sweeps on hyper-parameters. Configurations and values of hyperparameters for training, such as batch-size, learning rate, optimizers are stored in a yaml file ```src/dlip/conf/train_model.yaml```.

You can access the configuration & outputs of previously run scripts by default in the ```outputs``` folder.

More info on how to setup hydra for your projects [here](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b)

## Logging experiment

MLflow, Tensorboard is the most common Experiment Tracking solution .  We chose to use [MLflow](https://www.mlflow.org/).

## Code Packaging 

In python, the best way to load a module is to package it and install it.  There are several library for packaging code. Here we use [SetupTools](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)



After setting up your project, simply go to `\src` and launch

```pip install -r requirements.txt  install -e . ```

You will be able to load the package inside the notebook. 

[More info on packaging](https://packaging.python.org/en/latest/)

## README

A good README file provides all information for replication of the experiments. 

For research code, paperwithcode guidelines are a good source of information:
Check them [here](https://github.com/paperswithcode/releasing-research-code/tree/master)

General very useful guidelines for Python [here](https://docs.python-guide.org/) 


