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
classifier = 'SVM'
datasets = {
    # 'MUTAG': [1, 2, 4, 8, 16, 32],
    # 'PTC_MR': [1, 2, 4, 8, 16, 32, 64, 128],
    # 'NCI1': [1, 2, 4, 8, 16, 32],
    # 'PROTEINS_FULL': [1, 2, 4, 8, 16, 40],
    # 'DD': [1, 2, 4, 8, 16, 32, 64, 128, 256, 400],
    # 'COLLAB': [1, 2, 4, 8, 16, 32, 64, 128],
    # 'IMDB-BINARY': [1, 2, 4, 8, 16, 32],
    # 'IMDB-MULTI': [1, 2, 4, 8, 16],
    'REDDIT-BINARY': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    # 'REDDIT-MULTI-5K': [1, 2, 4, 8, 16, 32, 128, 256, 512, 780]
}

# parameters of MLP
epochs = 300
batch_size = 4
valid = 0.1
# For network robustness prediction
training_size = '(700,1300)'
testing_size = training_size
isd = 0
isw = 0
robustness = 'lc'
if robustness == 'yc':
    atk = 'nrnd'
else:
    atk = 'ndeg'
