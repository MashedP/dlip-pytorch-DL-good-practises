# Dlip TP1 - Introduction to Pytorch and practises
Welcome !

In this repository, you will find useful information  on good practises for Deep Learning research and developement projects.   

Note that those are general guidelines, recipes that evolves with the specificity of your project. We tend to prefer open-source solutions. 

This project will be in Python 3, using Pytorch. 

Have fun using this as a template and source of information for your first practical session & future projects !

*Inspired by [Solal Nathan's presentation](https://hebergement.universite-paris-saclay.fr/sepag/2023_05_24_Programming_Project_Management.pdf) on Programming project management*

# General information

LICENSE contains information about the license of your code. Companies may have guidelines on how to license code. 


# Collaborative Work 
### Git Versioning
.gitkeep files are super useful to commit an empty folder
.gitignore are useful to exclude elements from the versioning ( heavy datasets for example) 

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) Commits are traces of your changes on the project. Standards in commits leads to better communication between the actors of projects. Extensions exists for multiple projects 


### Cookie cutter information
[Cookie Cutter](https://drivendata.github.io/cookiecutter-data-science/) are templates of project that you can replicate and use as your own. They are great because their structure are familiar to other developper / ML engineer / Data Scientist. 


## Configuration

When we perform deep learning tasks, we often need to run multiple experiments 

[seeds] are important to assess reproducibility of your results. Learn how they work in pytorch [here](https://pytorch.org/docs/stable/notes/randomness.html).

requirements.txt is an important file for reproducibility. it contains all packages required to launch your experiment. 



[Hydra](https://hydra.cc/docs/intro/) is a python package to handle parsing and configuration files 
	- basé sur [OmegaConf](https://github.com/omry/omegaconf)

Package your code ! 
