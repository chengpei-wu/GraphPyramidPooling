import numpy as np
from parameters import *

for dataname in datasets:
    acc = np.load(f'./accuracy/GAT/{dataname}_nppr_10cv.npy')
    print(dataname, round(np.mean(acc) * 100, 3), round(np.std(acc) * 100, 3))
