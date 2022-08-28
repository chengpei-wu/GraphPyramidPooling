import networkx as nx
import numpy as np


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
    if has_node_attr:
        node_attr_vec = []
        for i in range(G.number_of_nodes()):
            node_attr_vec.append(G.nodes[i]['node_attr'])
        node_attr_vec = np.array(node_attr_vec).T
        for i in range(node_attr_vec.shape[0]):
            ranking_node_attr_vec = [node_attr_vec[i][k] for k in ranking_nodes_id]
            ranking_vec.append(ranking_node_attr_vec)

    for i in pooling_attr:
        if i == 'average_neighbor_degree':
            avn_degrees = list(nx.average_neighbor_degree(G).items())
            avn_degree_vec = [avn_degrees[k][1] for k in ranking_nodes_id]
            # avn_degree_vec = scaler.fit_transform(np.array(avn_degree_vec).reshape((len(avn_degree_vec), 1)))
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
            # degree_vec = scaler.fit_transform(np.array(degree_vec).reshape((len(degree_vec), 1)))
            ranking_vec.append(degree_vec)
        if i == 'clustering':
            clusterings = list(nx.clustering(G).items())
            clustering_vec = [clusterings[k][1] for k in ranking_nodes_id]
            ranking_vec.append(clustering_vec)
        if i == 'betweenness':
            betweennesses = list(nx.betweenness_centrality(G).items())
            betweenness_vec = [betweennesses[k][1] for k in ranking_nodes_id]
            ranking_vec.append(betweenness_vec)
        if i == 'h':
            features = np.array([G.nodes[i]['h'].detach().numpy() for i in range(G.number_of_nodes())])
            feature_vec = [features[k] for k in ranking_nodes_id]
            ranking_vec = feature_vec
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
    pooling_vec = []
    # for v in vec:
    #     for s in pooling_sizes:
    #         pooling_vec = np.concatenate([pooling_vec, pooling(v, s, pooling_way)])
    for s in pooling_sizes:
        for v in vec:
            pooling_vec = np.concatenate([pooling_vec, pooling(v, s, pooling_way)])
            # print(pooling_vec)
    return pooling_vec


def pooling(vec, size, pooling_way):
    single_size_pooling_vec = []
    length = len(vec)
    if length < size:
        while length < size:
            vec = np.append(vec, 0)
            length += 1
    divid = int(np.floor(length / size))
    for i in range(size):
        if pooling_way == 'max':
            single_size_pooling_vec.append(np.max(vec[i * divid:(i + 1) * divid]))
        elif pooling_way == 'mean':
            single_size_pooling_vec.append(np.mean(vec[i * divid:(i + 1) * divid]))
        else:
            single_size_pooling_vec.append(np.sum(vec[i * divid:(i + 1) * divid]))

    return single_size_pooling_vec
