from utils import load_data
from MLP import MLP
from parameters import *

path = f'{training_size}_{atk}_isd{isd}_isw{isw}'
x, y = load_data(
    path=f'./data/train/{path}.mat',
    isd=isd,
    roubustness=robustness,
    pooling_sizes=pooling_sizes
)

print(x.shape)

mlp = MLP(
    epochs=epochs,
    batch_size=batch_size,
    valid_proportion=valid,
    model=None,
    pooling_sizes=pooling_sizes
)
mlp.fit(x, y, f'./models/{path}_{robustness}')
