from evaluation_gnn import evaluation_gnn
from parameters import datasets

for dataset, pooling_sizes in zip(datasets.keys(), datasets.values()):
    for gnn_model in ['GraphSAGE']:
        for read_out in ['nppr', 'max', 'mean', 'min', 'sum']:
            evaluation_gnn(gnn_model, read_out, dataset, pooling_sizes, fold=10, times=10)
