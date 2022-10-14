# parameters of GraphPyramidPooling
rank_label = 'degree'
pooling_attr = [
    'degree',
    # 'closeness_centrality',
    # 'communicability_centrality',
    # 'harmonic_centrality',
    # 'page_rank',
    # 'triangles'
]
pooling_way = 'mean'
num_node_attr = 0

# For graph classification
classifier = 'GBDT'
datasets = {
    'IMDB-BINARY': [1, 2, 4, 8, 16, 32],
    'IMDB-MULTI': [1, 2, 4, 8, 16],
    'REDDIT-BINARY': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    'REDDIT-MULTI-5K': [1, 2, 4, 8, 16, 32, 128, 256, 512, 780],
    'MUTAG': [1, 2, 4, 8, 16, 32],
    'PTC_MR': [1, 2, 4, 8, 16, 32],
    'NCI1': [1, 2, 4, 8, 16, 32],
    'PROTEINS_FULL': [1, 2, 4, 8, 16, 40],
    'DD': [1, 2, 4, 8, 16, 32, 64, 128, 256, 400],
    'COLLAB': [1, 2, 4, 8, 16, 32, 64, 128],
}
