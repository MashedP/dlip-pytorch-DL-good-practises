# Dlip TP1 - Introduction to Pytorch and practises
Welcome !

In this repository, you will find useful information  on good practises for Deep Learning research and developement projects.   

Note that those are general guidelines, recipes that evolves with the specificity of your project. We tend to prefer open-source solutions. 

This project will be in Python 3, using Pytorch. 

Have fun using this as a template and source of information for your first practical session & future projects !

*Inspired by [Solal Nathan's presentation](https://hebergement.universite-paris-saclay.fr/sepag/2023_05_24_Programming_Project_Management.pdf) on Programming project management*

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
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
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
.gitignore are useful to exclude elements from the versioning ( heavy datasets for example) 

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) Commits are traces of your changes on the project. Standards in commits leads to better communication between the actors of projects. Extensions exists for multiple projects 

[Git CheatSheet](https://training.github.com/downloads/github-git-cheat-sheet.pdf)


## Configuration

When we perform deep learning tasks, we often need to run multiple experiments 

[seeds] are important to assess reproducibility of your results. Learn how they work in pytorch [here](https://pytorch.org/docs/stable/notes/randomness.html).

`requirements.txt` is an important file for reproducibility. It contains all packages required to launch your experiment. 


[Hydra](https://hydra.cc/docs/intro/) is a python package to handle parsing and configuration files 
	- based on [OmegaConf](https://github.com/omry/omegaconf)

## Code Packaging 

In python, the best way to load a module is to package it and install it.  There are several library for packaging code. Here we use [SetupTools](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)



After setting up your project, simply go to `\src` and launch

```pip install -r requirements.txt  install -e . ```

You will be able to load the package inside the notebook. 

[More info on packaging](https://packaging.python.org/en/latest/)