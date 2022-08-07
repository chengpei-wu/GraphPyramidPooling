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


def load_dgl_data(dataset, pooling_sizes, rank_label, pooling_attr, pooling_way):
    # for classification task (graph classification, real-world networks)
    enc = OneHotEncoder()
    data = TUDataset(dataset)
    num_classes = data.num_classes
    num_node_attr = 0
    x = []
    labels = []
    has_node_attr = 'node_attr' in data[0][0].nodes[0][0].keys()
    if has_node_attr:
        num_node_attr = len(data[0][0].nodes[0][0]['node_attr'].numpy().flatten())
        print(num_node_attr)
    for id in range(len(data)):
        print('\r',
              f'loading {id} / {len(data)}  network...',
              end='',
              flush=True)
        graph, label = data[id]
        # graph = dgl.to_simple_graph(graph)
        print(nx.weisfeiler_lehman_graph_hash(G))
        G = nx.DiGraph(dgl.to_networkx(graph))
        if has_node_attr:
            for i in range(G.number_of_nodes()):
                G.nodes[i]['node_attr'] = graph.nodes[i][0]['node_attr'].numpy().flatten()
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
    return x, y, num_classes, num_node_attr
