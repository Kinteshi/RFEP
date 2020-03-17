# Random Forest Evolutionary based Pruning

## Requirements

- Requires Python 3.7 and above
- Requires R to be installed
- `rpy2` Python package that can be found inside this repository
- `deap` Python package forked by Haiga in https://github.com/Haiga/deap

    > `pip install git+https://github.com/Haiga/deap#egg=deap` 

- `dill` Python package is necesary as well
- `sklearn`, `numpy`, `matplotlib` and `seaborn` are required as well

## Instructions


### Setup  


To use this model training package some assets are needed. Those are:

- A L2R  dataset:
    > Ex.: web10k, web30k, yahoo, 2003_td_dataset)
- Baselines for the specific dataset
- k-fold structured dataset

Each one of the above needs to be in its own folder. Like this:

>       data\
>           |dataset\
>           |       |dataset_name\
>           |
>           |baselines\
>           |         |dataset_name\

### Usage

The template file can be found in this repository under the name `template.py`. Pretty right-to-the-point.

#### Observations

This package is still under development.
