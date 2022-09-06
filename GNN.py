import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, GATConv
from GraphPyramidPooling import pyramid_pooling, node_ranking_by_label
import networkx as nx
from PP import PyramidPooling


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, readout, pyramid):
        super(GCN, self).__init__()
        self.readout = readout
        self.pyramid = pyramid
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=F.relu)
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=F.relu)
        if self.readout != 'nppr':
            self.input_layer = nn.Linear(hidden_dim, 512)
        else:
            self.nfpp_layer = PyramidPooling(self.pyramid)
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
        h = g.in_degrees().view(-1, 1).float()
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        num_per_batch = g.batch_num_nodes()
        node_degrees = g.in_degrees()
        if self.readout == 'min':
            hg = dgl.readout_nodes(g, feat='h', op='min')
        if self.readout == 'mean':
            hg = dgl.readout_nodes(g, feat='h', op='mean')
        if self.readout == 'max':
            hg = dgl.readout_nodes(g, feat='h', op='max')
        if self.readout == 'sum':
            hg = dgl.readout_nodes(g, feat='h', op='sum')
        if self.readout == 'nppr':
            hg = self.nfpp_layer(h, num_per_batch, node_degrees)
        return self.classify(hg)


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 aggregator_type,
                 readout,
                 pyramid):
        super(GraphSAGE, self).__init__()
        self.readout = readout
        self.pyramid = pyramid
        self.conv1 = SAGEConv(in_feats, n_hidden, aggregator_type, activation=F.relu)
        self.conv2 = SAGEConv(n_hidden, n_hidden, aggregator_type, activation=F.relu)
        if self.readout != 'nppr':
            self.input_layer = nn.Linear(n_hidden, 512)
        else:
            self.nfpp_layer = PyramidPooling(self.pyramid)
            self.input_layer = nn.Linear(n_hidden * sum(pyramid), 512)
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
        h = g.in_degrees().view(-1, 1).float()
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        num_per_batch = g.batch_num_nodes()
        node_degrees = g.in_degrees()
        if self.readout == 'min':
            hg = dgl.readout_nodes(g, feat='h', op='min')
        if self.readout == 'mean':
            hg = dgl.readout_nodes(g, feat='h', op='mean')
        if self.readout == 'max':
            hg = dgl.readout_nodes(g, feat='h', op='max')
        if self.readout == 'sum':
            hg = dgl.readout_nodes(g, feat='h', op='sum')
        if self.readout == 'nppr':
            hg = self.nfpp_layer(h, num_per_batch, node_degrees)
        return self.classify(hg)


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, readout, pyramid):
        super().__init__()
        self.readout = readout
        self.pyramid = pyramid
        self.conv1 = GATConv(in_size, hid_size, heads[0], activation=F.relu)
        self.conv2 = GATConv(hid_size * heads[0], hid_size, heads[1], activation=F.relu)
        if self.readout != 'nppr':
            self.input_layer = nn.Linear(hid_size, 512)
        else:
            self.nfpp_layer = PyramidPooling(self.pyramid)
            self.input_layer = nn.Linear(hid_size * sum(pyramid), 512)
        self.classify = nn.Sequential(
            self.input_layer,
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_size),
            nn.Softmax(),
        )

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        h = self.conv1(g, h)
        h = h.flatten(1)
        h = self.conv2(g, h)
        h = h.mean(1)
        g.ndata['h'] = h
        num_per_batch = g.batch_num_nodes()
        node_degrees = g.in_degrees()
        if self.readout == 'min':
            hg = dgl.readout_nodes(g, feat='h', op='min')
        if self.readout == 'mean':
            hg = dgl.readout_nodes(g, feat='h', op='mean')
        if self.readout == 'max':
            hg = dgl.readout_nodes(g, feat='h', op='max')
        if self.readout == 'sum':
            hg = dgl.readout_nodes(g, feat='h', op='sum')
        if self.readout == 'nppr':
            hg = self.nfpp_layer(h, num_per_batch, node_degrees)
        return self.classify(hg)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    loop_graphs = [dgl.add_self_loop(graph) for graph in graphs]
    return dgl.batch(loop_graphs), torch.tensor(labels, dtype=torch.long)


def node_embedding_pyramid_pooling(graph, pyramid):
    G = nx.DiGraph(dgl.to_networkx(graph, node_attrs='h'))
    vec = node_ranking_by_label(G, ['h'], 'degree')
    return pyramid_pooling(vec.T, pyramid, 'mean')
