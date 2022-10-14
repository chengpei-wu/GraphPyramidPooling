import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def graph2vec(G, rank_label, pooling_sizes, pooling_attr, pooling_way):
    ranking_vec = node_attr_ranking(G, rank_label, pooling_attr)
    pooling_vector = pyramid_pooling(ranking_vec, pooling_sizes, pooling_way)
    return pooling_vector


def node_attr_ranking(G, rank_label, pooling_attr):
    ranking_vec = node_ranking_by_label(G, pooling_attr, rank_label)
    return np.array(ranking_vec)


def node_ranking_by_label(G, pooling_attr, rank_label):
    if rank_label == 'random':
        ranking_nodes_id = np.random.permutation(G.number_of_nodes())
    elif rank_label == 'degree':
        ranking_nodes = sorted(nx.degree(G), key=lambda x: x[1], reverse=True)
        ranking_nodes_id = [n[0] for n in ranking_nodes]
    elif rank_label == 'betweenness':
        ranking_nodes = sorted(nx.betweenness_centrality(G).items(), key=lambda x: x[1], reverse=True)
        ranking_nodes_id = [n[0] for n in ranking_nodes]
    elif rank_label == 'unique':
        features = [[n[1]] for n in nx.degree(G)]
        for i in range(len(features)):
            bets = list(nx.betweenness_centrality(G).items())
            avngs = list(nx.average_neighbor_degree(G).items())
            features[i].append(avngs[i][1])
            features[i].append(bets[i][1])
        feature_dic = dict()
        for i in range(len(features)):
            feature_dic[f'{i}'] = features[i]
        feature_dic = sorted(feature_dic.items(), key=lambda x: (x[1][0], x[1][1], x[1][2]), reverse=True)
        ranking_nodes_id = [int(n[0]) for n in feature_dic]
    ranking_vec = []
    has_node_attr = 'node_attr' in G.nodes[0].keys()
    has_node_label = 'label' in G.nodes[0].keys()

    for i in pooling_attr:
        if i == 'average_neighbor_degree':
            avn_degrees = list(nx.average_neighbor_degree(G).items())
            avn_degree_vec = [avn_degrees[k][1] for k in ranking_nodes_id]
            ranking_vec.append(avn_degree_vec)
        if i == 'max_neighbor_degree':
            max_neighbor_degree_set = [np.max(n) for n in get_neighbor_degree_set(G)]
            maxn_degree_vec = [max_neighbor_degree_set[k] for k in ranking_nodes_id]
            ranking_vec.append(maxn_degree_vec)
        if i == 'min_neighbor_degree':
            min_neighbor_degree_set = [np.min(n) for n in get_neighbor_degree_set(G)]
            minn_degree_vec = [min_neighbor_degree_set[k] for k in ranking_nodes_id]
            ranking_vec.append(minn_degree_vec)
        if i == 'std_neighbor_degree':
            std_neighbor_degree_set = [np.std(n) for n in get_neighbor_degree_set(G)]
            stdn_degree_vec = [std_neighbor_degree_set[k] for k in ranking_nodes_id]
            ranking_vec.append(stdn_degree_vec)
        if i == 'degree':
            degrees = nx.degree(G)
            degree_vec = [degrees[k] for k in ranking_nodes_id]
            ranking_vec.append(degree_vec)
        if i == 'clustering':
            clusterings = list(nx.clustering(G).items())
            clustering_vec = [clusterings[k][1] for k in ranking_nodes_id]
            ranking_vec.append(clustering_vec)
        if i == 'eigenvector_centrality':
            eigenvector = list(nx.eigenvector_centrality(G).items())
            eigenvector_vec = [eigenvector[k][1] for k in ranking_nodes_id]
            ranking_vec.append(eigenvector_vec)
        if i == 'closeness_centrality':
            closeness = list(nx.closeness_centrality(G).items())
            closeness_vec = [closeness[k][1] for k in ranking_nodes_id]
            ranking_vec.append(closeness_vec)
        if i == 'communicability_centrality':
            centrality = list(nx.communicability_betweenness_centrality(G).items())
            centrality_vec = [centrality[k][1] for k in ranking_nodes_id]
            ranking_vec.append(centrality_vec)
        if i == 'harmonic_centrality':
            centrality = list(nx.harmonic_centrality(G).items())
            centrality_vec = [centrality[k][1] for k in ranking_nodes_id]
            ranking_vec.append(centrality_vec)
        if i == 'page_rank':
            centrality = list(nx.pagerank(G).items())
            centrality_vec = [centrality[k][1] for k in ranking_nodes_id]
            ranking_vec.append(centrality_vec)
        if i == 'triangles':
            centrality = list(nx.triangles(G).items())
            centrality_vec = [centrality[k][1] for k in ranking_nodes_id]
            ranking_vec.append(centrality_vec)

    return np.array(ranking_vec)


def get_neighbor_degree_set(G):
    nodes = G.nodes()
    degrees = nx.degree(G)
    neighbor_degree_set = [[degrees[i] for i in list(nx.neighbors(G, n))] for n in nodes]
    for i in range(len(neighbor_degree_set)):
        if not neighbor_degree_set[i]:
            neighbor_degree_set[i] = [0]
    return neighbor_degree_set


def pyramid_pooling(vec, pooling_sizes, pooling_way):
    all_pooling_vec = []
    for v in vec:
        pooling_vec = []
        for s in pooling_sizes:
            pooling_vec = np.concatenate([pooling_vec, pooling(v, s, pooling_way)])
        all_pooling_vec = np.concatenate([all_pooling_vec, pooling_vec])

    # pooling_vec = []
    # for s in pooling_sizes:
    #     for v in vec:
    #         pooling_vec = np.concatenate([pooling_vec, pooling(v, s, pooling_way)])
    return all_pooling_vec


def pooling(vec, size, pooling_way):
    length = len(vec)
    vec = torch.tensor(vec).view(1, length).float()
    kernel = int(math.ceil(length / size))
    pad1 = int(math.floor((kernel * size - length) / 2))
    pad2 = int(math.ceil((kernel * size - length) / 2))
    assert pad1 + pad2 == (kernel * size - length)
    padded_input = F.pad(input=vec, pad=[0, pad1 + pad2], mode='constant', value=0)
    if pooling_way == "max":
        pool = nn.MaxPool1d(kernel_size=kernel, stride=kernel, padding=0)
    else:
        pool = nn.AvgPool1d(kernel_size=kernel, stride=kernel, padding=0)
    x = pool(padded_input).flatten().numpy().tolist()
    return x
