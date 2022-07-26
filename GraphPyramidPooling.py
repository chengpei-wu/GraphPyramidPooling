import networkx as nx
import numpy as np
import scipy.io as sio


def graph2vec(G):
    ranking_vec = node_attr_ranking(G)
    pooling_vector = pyramid_pooling(ranking_vec)
    pass


def node_attr_ranking(G, label=None):
    ranking_vec = []
    if label is None:
        label = ['degree', 'betweenness']
    for i in label:
        if i == 'degree':
            ranking_vec.append(node_degree_ranking(G))
        if i == 'betweenness':
            ranking_vec.append(node_betweenness_ranking(G))
    # print(ranking_vec)
    return np.array(ranking_vec)


def node_degree_ranking(G):
    ranking_nodes = sorted(nx.degree(G), key=lambda x: x[1], reverse=True)
    ranking_degree = [n[1] for n in ranking_nodes]
    return np.array(ranking_degree)


def node_betweenness_ranking(G):
    betweenness = nx.betweenness_centrality(G)
    print(betweenness[0])
    node_betweenness = [i for i in betweenness]
    ranking_betweenness = sorted(node_betweenness, reverse=True)
    print(ranking_betweenness)
    return ranking_betweenness


def pyramid_pooling(vec, pooling_sizes=None):
    if pooling_sizes is None:
        pooling_sizes = [1, 2, 4, 8, 16]
    pass


def pooling(pooling_size, max_pooling=True):
    pass
