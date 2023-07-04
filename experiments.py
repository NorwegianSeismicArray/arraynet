# Copyright 2023 Andreas Koehler, Erik Myklebust, MIT license

import numpy as np
from train import run_experiment
from itertools import product

# Code for training ArrayNet

# base name of input data files
NAME = 'merged_arces_4Fre'
# input directories where data are located. Here you can add different data sets to train on.
datasets = ['data']
results = {}

# train models with and without sampling and class weights
#for cw,sw in product([True,False],[True,False]):
# no weighting (all models in published paper)
for cw,sw in product([False],[False]):
    for dataset in datasets:
        PATH = f'{dataset}/'

        X = np.load(PATH + f'X_{NAME}.npy')
        y_reg = np.load(PATH + f'y_reg_{NAME}.npy')
        y_cl = np.load(PATH + f'y_cl_{NAME}.npy')
        y_cl = np.array(list(map(lambda x: x.upper(), y_cl)))

        res = run_experiment(f'{NAME}_dataset_{dataset}_cw_{cw}_sw_{sw}', X, y_reg, y_cl, weight_class=cw, weight_angle=sw)

        results[f'{dataset}_cw_{cw}_sw_{sw}'] = res
        print(res)

print(results)

