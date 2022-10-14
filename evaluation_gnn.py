import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from GNNs import EarlyStopping
from GNNs import GCN, GraphSAGE, GAT, collate


def evaluation_gnn(gnn_model, readout, dataset, pooling_sizes, fold=10, times=10):
    data = dgl.data.TUDataset(dataset)
    n_classes = data.num_classes
    data = np.array([data[id] for id in range(len(data))], dtype=object)
    labels = np.array([g[1].numpy().tolist() for g in data])
    kf = StratifiedKFold(n_splits=fold, shuffle=True)
    scores = []
    if gnn_model == 'GCN':
        model = GCN(1, 16, n_classes, readout, pooling_sizes)
    elif gnn_model == 'GraphSAGE':
        model = GraphSAGE(1, 16, n_classes, 'mean', readout, pooling_sizes)
    else:
        model = GAT(1, 16, n_classes, [4, 1], readout, pooling_sizes)
    # model = model.cuda()
    for time in range(times):
        acc = []
        cv = 0
        for train_index, test_index in kf.split(data, labels):
            cv += 1
            early_stop = EarlyStopping(30, verbose=True,
                                       checkpoint_file_path=f'./checkpoints/{gnn_model}_checkpoint.pt')
            data_train, data_test = data[train_index], data[test_index]
            len_train = int(len(data_train) * 0.9)
            data_loader = DataLoader(data_train[:len_train], batch_size=256, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(data_test, batch_size=64, shuffle=False, collate_fn=collate)
            valid_loader = DataLoader(data_train[len_train:], batch_size=64, shuffle=False, collate_fn=collate)

            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            reduce_lr = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=5,
                threshold=1e-1,
                min_lr=1e-6,
                verbose=True
            )
            # 模型训练
            epoch_losses = []
            for epoch in range(200):
                model.train()
                epoch_loss = 0
                for iter, (batchg, label) in enumerate(data_loader):
                    # batchg, label = batchg.cuda(), label.cuda()
                    # loss_func = loss_func.cuda()
                    prediction = model(batchg)
                    loss = loss_func(prediction, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.detach().item()
                epoch_loss /= (iter + 1)
                print(f'{gnn_model}({readout}): {dataset}_times{time}_cv{cv}_epoch: {epoch}, loss {epoch_loss}')
                epoch_losses.append(epoch_loss)
                # early_stopping
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for it, (batchg, label) in enumerate(valid_loader):
                        # batchg, label = batchg.cuda(), label.cuda()
                        # loss_func = loss_func.cuda()
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
            model.load_state_dict(torch.load(f'./checkpoints/{gnn_model}_checkpoint.pt'))
            model.eval()
            test_pred, test_label = [], []
            with torch.no_grad():
                for it, (batchg, label) in enumerate(test_loader):
                    # batchg, label = batchg.cuda(), label.cuda()
                    pred = np.argmax(model(batchg), axis=1).tolist()
                    test_pred += pred
                    test_label += label.cpu().numpy().tolist()
            acc.append(accuracy_score(test_label, test_pred))
            print(f'TEST ACCURACY SCORE: {acc}')
        print(np.mean(acc), np.std(acc))
        scores.append(acc)
    np.save(f'./accuracy/{gnn_model}/{dataset}_{readout}_10cv', np.array(scores))


evaluation_gnn('GCN', 'nppr', 'MUTAG', [1, 2, 4, 8, 16], fold=10, times=10)
