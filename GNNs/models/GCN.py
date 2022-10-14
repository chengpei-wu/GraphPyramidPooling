import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from ..PyramidPooling import PyramidPooling


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
