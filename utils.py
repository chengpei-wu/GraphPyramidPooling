import networkx as nx
import numpy as np
import scipy.io as sio
from GraphPyramidPooling import graph2vec


def load_data(path, isd, roubustness, pooling_sizes):
    mat = sio.loadmat(path)
    len_net = len(mat['res'])
    len_instance = len(mat['res'][0])
    networks = mat['res']
    x = []
    y = []
    for i in range(len_net):
        for j in range(len_instance):
            print('\r',
                  f'loading {i * len_instance + j + 1} / {len_net * len_instance}  network...',
                  end='',
                  flush=True)
            adj = networks[i, j]['adj'][0, 0].todense()
            if isd:
                G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
            else:
                G = nx.from_numpy_matrix(adj, create_using=nx.Graph())
            x.append(graph2vec(G), pooling_sizes)
            if roubustness == 'lc':
                y.append(networks[i, j]['lc'][0][0])
            if roubustness == 'yc':
                y.append(networks[i, j]['yc'][0][0])
    x = np.array(x)
    y = np.array(y)
    return x, y
