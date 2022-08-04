# parameters of GraphPyramidPooling
pooling_sizes = [1, 2, 4, 8, 16, 32]
ranking_nodes_by = 'degree'
# pooling_attr = ['degree', 'neighbor_average_degree']
pooling_attr = ['degree', 'clustering']
max_pooling = True
num_node_attr = 0

# parameters of MLP
epochs = 300
batch_size = 4
valid = 0.1

# For grpah classification
datasets = [
    'MUTAG',
    'DD',
    'NCI1',
    'PTC_MR',
    'PROTEINS',
    'COLLAB',
    'IMDB-BINARY',
    'IMDB-MULTI',
    'REDDIT-BINARY',
    'REDDIT-MULTI-5K',
    # 'REDDIT-MULTI-12K'
]


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