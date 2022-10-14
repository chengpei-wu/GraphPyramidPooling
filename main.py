from evaluation_gnn import evaluation_gnn
from parameters import datasets

for dataset, pooling_sizes in zip(datasets.keys(), datasets.values()):
    for gnn_model in ['GCN', 'GAT', 'GraphSAGE']:
        for read_out in ['max', 'mean', 'min', 'sum', 'nppr']:
            evaluation_gnn(gnn_model, read_out, dataset, pooling_sizes, fold=10, times=10, allow_cuda=True)
