# github-search

### Repository for "Searching Github Python repositories with machine learning" masters thesis

#### Jakub Bartczuk

## Running this project

### Prerequisites

Preprocessing steps were tested on a machine with 64GB RAM.

For training Graph Neural Networks CUDA GPU is required.

### General remarks

The project uses `nbdev` to create Python files from Jupyter notebook. To "make" project run

``nbdev_build_lib; pip install -e .``

in the root directory.

We use [ploomber](https://github.com/ploomber/ploomber) for managing training and data preprocessing. 

For example to create csv files with extracted READMEs run

``ploomber build --partial make_readmes  --skip-upstream --force ``

Relevant definitions can be found in `pipeline.yaml` and `env.yaml`


### TODO Downloading data

### TODO Model checkpoints

### Training models

Ploomber step:

`run_gnn_experiment`

### TODO Using models

