import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
from GraphPyramidPooling import graph2vec
from sklearn.preprocessing import OneHotEncoder
from dgl.data import *
import torch
import os


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
    # for classification task (graph classification, real-world networks)
    enc = OneHotEncoder()
    data = TUDataset(dataset)
    num_classes = data.num_classes
    num_node_attr = 0
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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_file_path=''):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint_file_path = checkpoint_file_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f'valid loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.checkpoint_file_path}...')
        torch.save(model.state_dict(), self.checkpoint_file_path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
