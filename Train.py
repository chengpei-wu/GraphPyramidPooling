from utils import load_network
from MLP import MLP

x, y = load_network(path='./data/test/r(300,700)_nrnd_isd0_isw0.mat', isd=False, roubustness='yc')

mlp = MLP()
mlp.fit(x, y, './models/test_model')
