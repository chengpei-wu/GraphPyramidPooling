import numpy as np
from utils import print_progress, node_distribution
from sklearn.model_selection import StratifiedKFold, KFold

n = node_distribution('REDDIT-MULTI-5K')
n = sorted(n)
a = len(n)
print(a)
print(np.mean(n))
print(n[int(a * 0.8)])
