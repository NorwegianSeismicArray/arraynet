
import numpy as np
from train_sub import run_experiment
from itertools import product

FOLDS = 5

NAMES = ['norsar_arces','norsar_spits','helsinki_arces','merged_arces']

NAME = 'merged_arces_regional'

results = {}

for cw,sw in product([True,False],[True,False]):

    for dataset in ['data']: #['data','data_baf1_noaug','data_baf1','data_baf20','data_baf300']:

        PATH = f'/projects/active/MMON/Array_detection/ML_methods/csm_pattern_classification/tf/{dataset}/'
        #PATH = f'tf/{dataset}/'

        X = np.load(PATH + f'X_{NAME}.npy')
        y_reg = np.load(PATH + f'y_reg_{NAME}.npy')
        y_cl = np.load(PATH + f'y_cl_{NAME}.npy')
        y_cl = np.array(list(map(lambda x: x.upper(), y_cl)))

        res = run_experiment(f'{NAME}_dataset_{dataset}_cw_{cw}_sw_{sw}', X, y_reg, y_cl, tune=False, weight_class=cw, weight_angle=sw)

        results[f'{dataset}_cw_{cw}_sw_{sw}'] = res
        print(res)

print(results)

