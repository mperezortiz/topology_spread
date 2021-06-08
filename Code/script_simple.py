from itertools import product
from multiprocessing import Pool
from experiment import run_experiment

config_dict = {
    "network_family": 'powerlaw',
    "nodes": 500,
    "edges": 5,
    "p": 0.0,
    "budget": 'unequal'
}

run_experiment(config_dict, 1)
