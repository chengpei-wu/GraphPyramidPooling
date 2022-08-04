import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from parameters import pooling_attr, ranking_nodes_by, max_pooling, num_node_attr


def graph2vec(G, pooling_sizes):
    ranking_vec = node_attr_ranking(G)
    pooling_vector = pyramid_pooling(ranking_vec, pooling_sizes)
    return pooling_vector


def node_attr_ranking(G, label=ranking_nodes_by):
    ranking_vec = []
    if label == 'degree':
        ranking_vec = node_degree_ranking(G)
    return np.array(ranking_vec)


def node_degree_ranking(G, attr=pooling_attr):
    scaler = MinMaxScaler()
    scaler = MinMaxScaler()
    ranking_nodes = sorted(nx.degree(G), key=lambda x: x[1], reverse=True)
    ranking_nodes_id = [n[0] for n in ranking_nodes]
    ranking_vec = []
    has_node_attr = 'node_attr' in G.nodes[0].keys()
    if has_node_attr:
        node_attr_vec = []
        for i in range(G.number_of_nodes()):
            node_attr_vec.append(G.nodes[i]['node_attr'])
        node_attr_vec = np.array(node_attr_vec).T
        # print(node_attr_vec.shape)
        for i in range(node_attr_vec.shape[0]):
            ranking_node_attr_vec = [node_attr_vec[i][k] for k in ranking_nodes_id]
            ranking_vec.append(ranking_node_attr_vec)
    for i in attr:
        if i == 'average_neighbor_degree':
            avn_degrees = list(nx.average_neighbor_degree(G).items())
            avn_degree_vec = [avn_degrees[k][1] for k in ranking_nodes_id]
            ranking_vec.append(avn_degree_vec)

        if i == 'degree':
            degrees = nx.degree(G)
            degree_vec = [degrees[k] for k in ranking_nodes_id]
            degree_vec = scaler.fit_transform(np.array(degree_vec).reshape((len(degree_vec), 1)))
            ranking_vec.append(list(degree_vec.flatten()))
        if i == 'clustering':
            clusterings = list(nx.clustering(G).items())
            clustering_vec = [clusterings[k][1] for k in ranking_nodes_id]
            ranking_vec.append(clustering_vec)
        if i == 'betweenness':
            betweennesses = list(nx.betweenness_centrality(G).items())
            betweenness_vec = [betweennesses[k][1] for k in ranking_nodes_id]
            ranking_vec.append(betweenness_vec)
        if i == 'closeness':
            closenesses = list(nx.closeness_centrality(G).items())
            closeness_vec = [closenesses[k][1] for k in ranking_nodes_id]
            ranking_vec.append(closeness_vec)
    return np.array(ranking_vec)


def pyramid_pooling(vec, pooling_sizes=None):
    if pooling_sizes is None:
        pooling_sizes = [1, 2, 4, 8, 16]
    pooling_vec = []
    for v in vec:
        for s in pooling_sizes:
            pooling_vec = np.concatenate([pooling_vec, pooling(v, s, max_pooling)])
    return pooling_vec


def pooling(vec, size, max_pooling=True):
    single_size_pooling_vec = []
    length = len(vec)
    if length < size:
        while length < size:
            vec = np.append(vec, 0)
            length += 1
    divid = int(np.floor(length / size))
    for i in range(size):
        if max_pooling:
            single_size_pooling_vec.append(np.max(vec[i * divid:(i + 1) * divid]))
        else:
            single_size_pooling_vec.append(np.mean(vec[i * divid:(i + 1) * divid]))
    return single_size_pooling_vec
