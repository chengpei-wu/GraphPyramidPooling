import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from ..PyramidPooling import PyramidPooling


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, readout, pyramid):
        super().__init__()
        self.readout = readout
        self.pyramid = pyramid
        self.conv1 = GATConv(in_size, hid_size, heads[0], activation=F.relu)
        self.conv2 = GATConv(hid_size * heads[0], hid_size, heads[1], activation=F.relu)
        if self.readout != 'nppr':
            self.input_layer = nn.Linear(hid_size, out_size)
        else:
            self.nfpp_layer = PyramidPooling(self.pyramid)
            self.input_layer = nn.Linear(hid_size * sum(pyramid), out_size)
        self.classify = nn.Sequential(
            self.input_layer,
            nn.Dropout(0.5)
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
