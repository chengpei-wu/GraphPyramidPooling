import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
from GraphPyramidPooling import graph2vec
from sklearn.preprocessing import OneHotEncoder
from dgl.data import *


def load_data(path, isd, roubustness, pooling_sizes):
    # for regression task (robustness prediction)
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
            x.append(graph2vec(G, pooling_sizes))
            if roubustness == 'lc':
                y.append(networks[i, j]['lc'][0][0])
            if roubustness == 'yc':
                y.append(networks[i, j]['yc'][0][0])
    x = np.array(x)
    y = np.array(y)
    return x, y


def load_dgl_data(pooling_sizes, dataset='REDDIT-BINARY'):
    # for classification task (graph classification, real-world networks)
    enc = OneHotEncoder()
    data = TUDataset(dataset)
    x = []
    labels = []
    adj0 = data[0][0].adjacency_matrix().to_dense().numpy()
    if np.sum(adj0.T != adj0):
        isd = 1
    else:
        isd = 0
    for id in range(len(data)):
        print('\r',
              f'loading {id} / {len(data)}  network...',
              end='',
              flush=True)
        graph, label = data[id]
        adj = graph.adjacency_matrix().to_dense().numpy()
        if isd:
            G = nx.from_numpy_matrix(adj, create_using=nx.DiGraph())
        else:
            G = nx.from_numpy_matrix(adj, create_using=nx.Graph())
        x.append(graph2vec(G, pooling_sizes))
        labels.append(label.numpy())
    y = enc.fit_transform(labels).toarray()
    return x, y
