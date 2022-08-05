import numpy as np
from parameters import datasets

for dataset in datasets:
    print('{:^15}'.format(dataset),end=': ')
    for mod in ['SVM']:
        acc = np.load(f'./accuracy/{mod}/{dataset}_cv.npy')
        print('{:^6}'.format(round(np.max(np.mean(acc,axis=1))*100, 2)), end=f'({mod})  ')
    print()