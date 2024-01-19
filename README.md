# Dlip TP1 - Introduction to Pytorch and good practises
Welcome !

In this repository, you will find useful information on good practices for Deep Learning and Data Science research and development projects.

Note that those are general guidelines, recipes that evolves with the specificity of your project. The overall philosophy is "start small and incrementally add tools to understand what you are doing".

We prefer to use open-source solutions. 

This project will be in Python 3, using Pytorch. 

Have fun using this as a template and source of information for your first practical session & future projects !

This template already includes code from the first exercise of TP1. Try to adapt it for the second exercise.

## Project Checklist

- [x] Packaging
- [x] Hydra for configuration management
- [x] MLflow for experiment tracking
- [x] PyTorch Trainer class
- [x] PyTorch Model class
- [x] Model Evaluation
- [x] Model checkpointing (saving and loading)
- [ ] Comparative study of multiple models


## Project Structure

Below is the recommended project structure, enabling a modular and manageable codebase.

```
├── LICENSE            <- Information about the license of your code. Companies may have guidelines on how to license code. [See more here](https://choosealicense.com/)
├── README.md          <-  A README file for developers to understand the project setup and instructions.
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
│   └── figures        <- Graphics and figures for use in reports.
│ 
├── requirements.txt   <- Required libraries and dependencies. 
│
├── pyproject.toml     <- Make the project pip installable with `pip install -e`.
├── src/dlip           <- Source code of the project. `dlip` is the name of the package, you will import it using `import dlip`
│   ├── __init__.py    <- Initializes the 'dlip' Python package.
│   │
│   ├── conf           <- Configuration files for experiments (YAML files managed by Hydra).
│   │   └── train_linear.yaml
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models, to use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py

```

This structure is inspired by [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

 Cookie Cutters are templates of projects you can replicate and use as your own. They are great because their structure are familiar to other developper / ML engineer / Data Scientist. 

## Collaborative Work

### Version Control with Git
- `.gitkeep` files allow you to commit empty folders.
- `.gitignore` files exclude certain files from version control (like large datasets). Find templates [here](https://github.com/github/gitignore).
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) promote standard commit messages, facilitating better team communication. Plugins exists for your IDE. 
- [Git Cheat Sheet](https://training.github.com/downloads/github-git-cheat-sheet.pdf) is useful for quick command reference.

### Contributing
To contribute to this project, look at the Issues section on GitLab.

Issues are here to communicate with the development team about suggestions, bugs or general questions. 

Feel free to fork this repo and create a merge request with your proposed changes, and remember that we are all here to learn, even from mistakes.

## Best Practices and Tool Usage for Data Science Projects

### Seeds 

Using *seeds* is important to assess reproducibility of your results. Learn how to manage randomness in PyTorch [here](https://pytorch.org/docs/stable/notes/randomness.html).

### Argument Parsing with Hydra

Project parameters can be efficiently managed with [Hydra](https://hydra.cc/docs/intro/), greatly simplifying the process of running experiments with various configurations.

Common practise is to use a parser in order to modify parameters you are using from the command line. In this repository, we use the Hydra package to handle arguments & configurations.

Hydra allows you to parse arguments from the command line, to launch and log multiple experiments with different configuration easily. 

To run the training script, you can for example run 

```python src/dlip/models/train_model.py```

or with a different batch size :

```python src/dlip/models/train_model.py train.batch_size=20```

or launching multiple experiment on various batch sizes :

```python src/dlip/models/train_model.py --multirun train.batch_size=10,20,30,40,50```

It helps when you have to perform sweeps on hyperparameters. 

Default Configurations and values of hyperparameters such as batch-size, learning rate, optimizers are stored in a yaml file ```src/dlip/conf/train_model.yaml```

You can access the configuration & outputs of previously run scripts by default in the ```outputs``` folder.

For details on setting up and using Hydra in your projects, [see here](https://towardsdatascience.com/complete-tutorial-on-how-to-use-hydra-in-machine-learning-projects-1c00efcc5b9b).


### Experiment Logging with MLflow
For tracking experiments, packages like Tensorboard,W&B or Aim exists.  MLflow is our open-source choice. To integrate it smoothly with Hydra, you can follow guidelines in this [blogpost](https://medium.com/optuna/easy-hyperparameter-management-with-hydra-mlflow-and-optuna-783730700e7d). Launch the MLflow user interface via `mlflow ui` in your terminal and view it in your browser to monitor the experiments.


## Code Packaging 

In python, the best way to load a module is to package it and install it.  There are several library for packaging code. Here we use [SetupTools](https://setuptools.pypa.io/en/latest/userguide/quickstart.html)

`requirements.txt` is an important file for reproducibility. It contains all packages required to launch your experiment. 

To package the project,  go to the root of the project and launch

```pip install -r requirements.txt  install -e . ```

You will be able to load the package inside the notebook. 

[More info on packaging](https://packaging.python.org/en/latest/)


### Environment Management
It is recommended to build a new environment for each of your different project. 
However, some modules like Pytorch are taking a lot of memory. 

 - [Virtual Env](https://realpython.com/python-virtual-environments-a-primer/)
 - [Conda](https://www.anaconda.com/download/)

 A more advanced way for reproducibility of your code is to contenerize your application, with tools like Docker. 


### Python Coding Practices
Refer to the Hitchhiker's Guide to Python for industry-standard guidelines on writing clean and maintainable Python code. You can find it [here](https://docs.python-guide.org/).

### Further Reading
- For research-focused projects, "Papers with Code" provides excellent guidelines available [here](https://github.com/paperswithcode/releasing-research-code).

Utilize the above structure and guidelines as a starting point and adapt them as necessary to fit your project's unique requirements. Happy coding!

*Inspired by [Solal Nathan's presentation](https://hebergement.universite-paris-saclay.fr/sepag/2023_05_24_Programming_Project_Management.pdf) on programming project management.*