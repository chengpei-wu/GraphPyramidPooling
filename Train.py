from utils import load_data
from MLP import MLP
from parameters import *

x, y = load_data(
    path='./data/test/r(300,700)_nrnd_isd0_isw0.mat',
    isd=False,
    roubustness='yc',
    pooling_sizes=pooling_sizes
)

mlp = MLP(
    epochs=epochs,
    batch_size=batch_size,
    valid_proportion=valid,
    model=None,
    pooling_sizes=pooling_sizes
)
mlp.fit(x, y, './models/test_model')
