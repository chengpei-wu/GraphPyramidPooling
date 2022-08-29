# parameters of GraphPyramidPooling
pooling_sizes = [1, 2, 4, 8, 16, 32, 64, 256, 780]
rank_label = 'degree'
pooling_attr = [
    'degree',
    'average_neighbor_degree',
    'max_neighbor_degree',
    'min_neighbor_degree',
    'std_neighbor_degree',
]
pooling_way = 'mean'
num_node_attr = 0

# For graph classification
classifier = 'RF'
datasets = [
    # 'MUTAG',
    # 'DD',
    # 'NCI1',
    # 'PTC_MR',
    # 'PROTEINS_FULL',
    # 'COLLAB',
    # 'IMDB-BINARY',
    # 'IMDB-MULTI',
    # 'REDDIT-BINARY',
    'REDDIT-MULTI-5K'
]

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
