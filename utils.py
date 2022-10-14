import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
from dgl.data import *
from sklearn.preprocessing import OneHotEncoder

from GraphPyramidPooling import graph2vec


def node_distribution(dataset):
    data = TUDataset(dataset)
    node_numbers = []
    for i in range(len(data)):
        node_numbers.append(len(data[i][0].adjacency_matrix()))
    return node_numbers


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


def load_dgl_data(dataset, pooling_sizes, rank_label, pooling_attr, pooling_way):
    enc = OneHotEncoder()
    data = TUDataset(dataset)
    x = []
    labels = []
    has_node_attr = 'node_attr' in data[0][0].nodes[0][0].keys()
    has_node_label = 'node_labels' in data[0][0].nodes[0][0].keys()
    if has_node_attr:
        num_node_attr = len(data[0][0].nodes[0][0]['node_attr'].numpy().flatten())
        print(f'Number of Node Attributes: {num_node_attr}')
    if has_node_label:
        print('Graph Has Node Labels')
    for id in range(len(data)):
        print('\r',
              f'loading {id} / {len(data)}  network...',
              end='',
              flush=True)
        graph, label = data[id]
        G = nx.Graph(dgl.to_networkx(graph))
        if has_node_attr:
            for i in range(G.number_of_nodes()):
                G.nodes[i]['node_attr'] = graph.nodes[i][0]['node_attr'].numpy().flatten()
        if has_node_label:
            for i in range(G.number_of_nodes()):
                G.nodes[i]['label'] = graph.nodes[i][0]['node_labels'].numpy().flatten()
        x.append(
            graph2vec(
                G=G,
                rank_label=rank_label,
                pooling_sizes=pooling_sizes,
                pooling_attr=pooling_attr,
                pooling_way=pooling_way
            )
        )
        labels.append(label.numpy())
    y = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    x = np.array(x)
    return x, y


def print_progress(now, total, length=20, prefix='progress:'):
    print('\r' + prefix + ' %.2f%%\t' % (now / total * 100), end='')
    print('[' + '>' * int(now / total * length) + '-' * int(length - now / total * length) + ']', end='')
