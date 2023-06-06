# Community Detection Through a New Prism: Bridging Archetypal Analysis and Stochastic Block Models

This repo contains all the code used in hhtps://arxiv.org/

## Installation

To download the code and install the dependencies, run the following commands:

```bash
git clone git@github.com:aleixalcacer/JA-BIAA-SBM.git
cd JA-BIAA-SBM
pip install -r requirements.txt
```

## Structure

The code is structured as follows:

- `datasets/`: contains the datasets used in the paper.
- `experiments/`: contains the code used to run the experiments.
- `figures/`: contains the code used to generate the figures.
- `models/`: contains the source code of the project.

## Run scripts

To run the experiments in a remote server, run the following command:

```shell
PYTHONPATH=. nohup python experiments/script_datasets.py {dataset} > {dataset}.log &
```
