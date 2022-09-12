import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from GNN import collate
from utils import EarlyStopping
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from parameters import datasets
from PP import PyramidPooling


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, readout, pyramid):
        super().__init__()
        self.layers = nn.ModuleList()
        self.pyramid = pyramid
        self.readout = readout
        # two-layer GCN
        self.conv1 = dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        self.conv2 = dglnn.GraphConv(hid_size, hid_size)
        if readout == 'nppr':
            self.input_layer = nn.Linear(hid_size * sum(pyramid), 512)
        else:
            self.input_layer = nn.Linear(hid_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.nfpp_layer = PyramidPooling(pyramid)
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
        h = self.dropout(h)
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


for readout in ['max']:
    for data_name, pooling_sizes in zip(datasets.keys(), datasets.values()):
        data = dgl.data.TUDataset(data_name)
        n_classes = data.num_classes
        data = np.array([data[id] for id in range(len(data))], dtype=object)
        labels = np.array([g[1].numpy().tolist() for g in data])
        # print(labels)
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        acc = []
        cv = 0
        for train_index, test_index in kf.split(data, labels):
            cv += 1
            early_stop = EarlyStopping(15, verbose=True, checkpoint_file_path=f'./test_checkpoint.pt')
            data_train, data_test = data[train_index], data[test_index]
            len_train = int(len(data_train) * 0.9)
            data_loader = DataLoader(data_train[:len_train], batch_size=256, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(data_test, batch_size=64, shuffle=False, collate_fn=collate)
            valid_loader = DataLoader(data_train[len_train:], batch_size=64, shuffle=False, collate_fn=collate)

            print('xxx')
            # 构造模型
            model = GCN(1, 16, n_classes, readout, pooling_sizes)
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=5,
                threshold=1e-5,
                min_lr=1e-6,
                verbose=True
            )
            # 模型训练
            epoch_losses = []
            for epoch in range(100):
                model.train()
                epoch_loss = 0
                for iter, (batchg, label) in enumerate(data_loader):
                    prediction = model(batchg)
                    loss = loss_func(prediction, label)
                    optimizer.zero_grad()
                    loss.backward()
                    # print(model.conv2)
                    # for name, parms in model.named_parameters():
                    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                    #           ' -->grad_value:', parms.grad)
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                epoch_loss /= (iter + 1)
                print(f'GCN({readout})----{data_name}_cv({cv})_epoch: {epoch}, loss {epoch_loss}')
                epoch_losses.append(epoch_loss)

                # early_stopping
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for it, (batchg, label) in enumerate(valid_loader):
                        prediction = model(batchg)
                        loss = loss_func(prediction, label)
                        valid_loss += loss.detach().item()
                    valid_loss /= (it + 1)
                reduce_lr.step(valid_loss)
                early_stop(valid_loss, model)
                if early_stop.early_stop:
                    print("Early stopping")
                    break

            # 测试
            # model.load_state_dict(torch.load(f'{gnn_model}_checkpoint.pt'))
            model.eval()
            test_pred, test_label = [], []
            with torch.no_grad():
                for it, (batchg, label) in enumerate(test_loader):
                    # batchg, label = batchg.to(DEVICE), label.to(DEVICE)
                    pred = np.argmax(model(batchg), axis=1).tolist()
                    test_pred += pred
                    test_label += label.cpu().numpy().tolist()
            print("Test accuracy: ", accuracy_score(test_label, test_pred))
            acc.append(accuracy_score(test_label, test_pred))
        print(np.mean(acc), np.std(acc))
        # np.save(f'./accuracy/{gnn_model}/{data_name}_{readout}_10cv', np.array(acc))
