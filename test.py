import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
from GraphPyramidPooling import graph2vec
from sklearn.preprocessing import OneHotEncoder
from dgl.data import *


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
    for id in range(1):
        print('\r',
              f'loading {id} / {len(data)}  network...',
              end='',
              flush=True)
        graph, label = data[id]
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


x, y, _, _ = load_dgl_data(
    dataset='MUTAG',
    pooling_sizes=[1, 2, 4, 8],
    rank_label='degree',
    pooling_attr=['degree', 'average_neighbor_degree'],
    pooling_way='max'
)
print(x)
