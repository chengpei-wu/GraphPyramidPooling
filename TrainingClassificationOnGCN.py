import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import accuracy_score
from GNN import GCN, GraphSAGE, GAT, collate
from utils import EarlyStopping
from parameters import datasets

gnn_model = 'GCN'
for readout in ['nppr']:
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
            early_stop = EarlyStopping(15, verbose=True, checkpoint_file_path=f'./{gnn_model}_checkpoint.pt')
            data_train, data_test = data[train_index], data[test_index]
            len_train = int(len(data_train) * 0.9)
            data_loader = DataLoader(data_train[:len_train], batch_size=256, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(data_test, batch_size=64, shuffle=False, collate_fn=collate)
            valid_loader = DataLoader(data_train[len_train:], batch_size=64, shuffle=False, collate_fn=collate)

            print('xxx')
            # 构造模型
            if gnn_model == 'GCN':
                model = GCN(1, 16, n_classes, readout, pooling_sizes)
            elif gnn_model == 'GraphSAGE':
                model = GraphSAGE(1, 16, n_classes, 'mean', readout, pooling_sizes)
            else:
                model = GAT(1, 16, n_classes, [4, 1], readout, pooling_sizes)
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
                print(f'{gnn_model}({readout})----{data_name}_cv({cv})_epoch: {epoch}, loss {epoch_loss}')
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
        np.save(f'./accuracy/{gnn_model}/{data_name}_{readout}_10cv', np.array(acc))
