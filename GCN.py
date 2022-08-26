import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
from GraphPyramidPooling import pyramid_pooling, node_ranking_by_label
import networkx as nx


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, readout, pyramid):
        super(Classifier, self).__init__()
        self.readout = readout
        self.pyramid = pyramid
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        if self.readout != 'nppr':
            self.input_layer = nn.Linear(hidden_dim, 512)
        else:
            self.input_layer = nn.Linear(hidden_dim * sum(pyramid), 512)
        self.classify = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            nn.Softmax(),
        )

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()  # [N, 1]
        h = F.relu(self.conv1(g, h))  # [N, hidden_dim]
        h = F.relu(self.conv2(g, h))  # [N, hidden_dim]
        g.ndata['h'] = h
        if self.readout == 'min':
            hg = dgl.readout_nodes(g, feat='h', op='min')
        if self.readout == 'mean':
            hg = dgl.readout_nodes(g, feat='h', op='mean')
        if self.readout == 'max':
            hg = dgl.readout_nodes(g, feat='h', op='max')
        if self.readout == 'sum':
            hg = dgl.readout_nodes(g, feat='h', op='sum')
        if self.readout == 'nppr':
            graphs = dgl.unbatch(g)
            hg = []
            for graph in graphs:
                G = nx.DiGraph(dgl.to_networkx(graph, node_attrs='h'))
                vec = node_ranking_by_label(G, ['h'], 'degree')
                hg.append(pyramid_pooling(vec.T, [1, 2, 4, 8, 16], 'mean'))
            hg = torch.tensor(np.array(hg)).to(torch.float32)
        return self.classify(hg)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(labels, dtype=torch.long)
