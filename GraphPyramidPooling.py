import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def graph2vec(G):
    ranking_vec = node_attr_ranking(G)
    pooling_vector = pyramid_pooling(ranking_vec)
    return pooling_vector


def node_attr_ranking(G, label=None):
    ranking_vec = []
    if label is None:
        label = ['degree', 'betweenness', 'clustering']
    for i in label:
        if i == 'degree':
            ranking_vec.append(node_degree_ranking(G))
        if i == 'betweenness':
            ranking_vec.append(node_betweenness_ranking(G))
        if i == 'clustering':
            ranking_vec.append(node_clustering_ranking(G))
    return np.array(ranking_vec)


def node_degree_ranking(G):
    scaler = MinMaxScaler()
    ranking_nodes = sorted(nx.degree(G), key=lambda x: x[1], reverse=True)
    ranking_degree = [n[1] for n in ranking_nodes]
    ranking_degree = scaler.fit_transform(np.array(ranking_degree).reshape((len(ranking_degree), 1)))
    return ranking_degree.flatten()


def node_betweenness_ranking(G):
    scaler = MinMaxScaler()
    ranking_nodes = sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)
    ranking_betweenness = [n[1] for n in ranking_nodes]
    ranking_betweenness = scaler.fit_transform(np.array(ranking_betweenness).reshape((len(ranking_betweenness), 1)))
    return ranking_betweenness.flatten()


def node_clustering_ranking(G):
    ranking_nodes = sorted(nx.clustering(G).items(), key=lambda x: x[1], reverse=True)
    ranking_clustering = [n[1] for n in ranking_nodes]
    return ranking_clustering


def pyramid_pooling(vec, pooling_sizes=None):
    if pooling_sizes is None:
        pooling_sizes = [1, 2, 4, 8, 16]
    pooling_vec = []
    for v in vec:
        for s in pooling_sizes:
            pooling_vec = np.concatenate([pooling_vec, pooling(v, s)])
    return pooling_vec


def pooling(vec, size, max_pooling=True):
    single_size_pooling_vec = []
    length = len(vec)
    divid = int(np.floor(length / size))
    for i in range(size):
        if max_pooling:
            single_size_pooling_vec.append(np.max(vec[i * divid:(i + 1) * divid]))
        else:
            single_size_pooling_vec.append(np.mean(vec[i * divid:(i + 1) * divid]))
    return single_size_pooling_vec
