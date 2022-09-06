import numpy as np
from parameters import datasets

for dataset in datasets:
    print(f'r-{dataset:^15}', end=': ')
    for mod in ['MLP']:
        acc = np.load(f'./accuracy/{mod}/{dataset}_random_10cv.npy')
        print(fr'{round(np.mean(acc) * 100, 2):^6}({round(np.std(acc) * 100, 2)})', end=f'({mod})  ')
    print()
for dataset in datasets:
    print(f'd-{dataset:^15}', end=': ')
    for mod in ['MLP']:
        acc = np.load(f'./accuracy/{mod}/{dataset}_degree_10cv.npy')
        print(fr'{round(np.mean(acc) * 100, 2):^6}({round(np.std(acc) * 100, 2)})', end=f'({mod})  ')
    print()
