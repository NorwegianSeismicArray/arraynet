

import pickle
import numpy as np
from tqdm import tqdm

catalog = 'norsar'
array = 'arces'
outdir = '/nobackup/erik/data/'

offset = 60
sr = 40

def filter_labels(label):
    if 'S' in label:
        return 'S'
    elif 'P' in label:
        return 'P'
    elif 'L' in label:
        return 'S'
    else:
        return 'N'

def angle_complex(angle):
    if angle is None:
        return 0,0
    angle = np.deg2rad(angle)
    return np.cos(angle), np.sin(angle)

with open(outdir+"traindata_cohthr_"+catalog+"_"+array+".p","rb") as f:
    data = pickle.load(f)

x = np.asarray(data['data'])
times = np.asarray(data['arrival'])
arrival_index = np.ones(len(x)) * offset * sr
cl = np.asarray(list(map(filter_labels, data['label'])))
y = np.asarray(list(map(angle_complex, data['baz'])))

np.save(f'{outdir}X_{catalog}_{array}.npy',x)
np.save(f'{outdir}ycl_{catalog}_{array}.npy',cl)
np.save(f'{outdir}yreg_{catalog}_{array}.npy',y)
np.save(f'{outdir}t_{catalog}_{array}.npy',times)
np.save(f'{outdir}idx_{catalog}_{array}.npy',arrival_index)