from utils import load_data
from MLP import MLP
from parameters import *
import scipy.io as sio
import time

path = f'{testing_size}_{atk}_isd{isd}_isw{isw}'

mlp = MLP(
    model=f'./models/{path}_{robustness}.hdf5',
    pooling_sizes=pooling_sizes
)

t1 = time.time()
x, y = load_data(
    path=f'./data/test/{path}.mat',
    isd=isd,
    roubustness=robustness,
    pooling_sizes=pooling_sizes
)
print(x.shape)

pred = mlp.my_predict(x)
t2 = time.time()

print(f'mean time: {(t2 - t1) / 900}')

sio.savemat(f'./prediction/{path}_{robustness}.mat', {'pred': pred, 'sim': y})
