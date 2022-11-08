import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from GNNs import GCN, GraphSAGE, GAT, collate, EarlyStopping
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def eval(model, data_loader, device):
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(data_loader):
            batchg, label = batchg.to(device), label.to(device)
            pred = np.argmax(model(batchg).cpu(), axis=1).tolist()
            test_pred += pred
            test_label += label.cpu().numpy().tolist()
    acc = accuracy_score(test_label, test_pred)
    return acc


def valid(model, valid_loader, early_stop, device):
    valid_acc = eval(model, valid_loader, device)
    early_stop(valid_acc, model)
    return valid_acc


def train(model, data_loader, valid_loader, epoches, device, gnn_model, readout, time, cv, dataset):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    early_stop = EarlyStopping(
        epoches // 2,
        verbose=True,
        checkpoint_file_path=f'./checkpoints/{gnn_model}_checkpoint.pt'
    )
    # 模型训练
    epoch_acc = []
    for epoch in range(epoches):
        model.train()
        for iter, (batchg, label) in enumerate(data_loader):
            batchg, label = batchg.to(device), label.to(device)
            loss_func = loss_func.to(device)
            prediction = model(batchg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_acc = eval(model, data_loader, device)
        valid_acc = valid(model, valid_loader, early_stop, device)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(
            f'{gnn_model}({readout}): {dataset}_times{time}_cv{cv}_epoch: {epoch}, lr: {lr}, acc: {train_acc}, val_acc: {valid_acc}')
        epoch_acc.append(train_acc)
        if early_stop.early_stop:
            print("Early stopping")
            break


def evaluation_gnn(gnn_model, readout, dataset, pooling_sizes, fold=10, times=10, epoches=120, allow_cuda=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dgl.data.TUDataset(dataset)
    n_classes = data.num_classes
    data = np.array([data[id] for id in range(len(data))], dtype=object)
    labels = np.array([g[1].numpy().tolist() for g in data])
    kf = StratifiedKFold(n_splits=fold, shuffle=True)
    scores = []

    for time in range(times):
        cv = 0
        for train_index, test_index in kf.split(data, labels):
            cv += 1
            data_train, data_test = data[train_index], data[test_index]
            len_train = int(len(data_train) * 0.9)
            data_loader = DataLoader(data_train[:len_train], batch_size=256, shuffle=True, collate_fn=collate)
            test_loader = DataLoader(data_test, batch_size=64, shuffle=False, collate_fn=collate)
            valid_loader = DataLoader(data_train[len_train:], batch_size=256, shuffle=False, collate_fn=collate)
            if gnn_model == 'GCN':
                model = GCN(1, 16, n_classes, readout, pooling_sizes)
            elif gnn_model == 'GraphSAGE':
                model = GraphSAGE(1, 16, n_classes, 'mean', readout, pooling_sizes)
            else:
                model = GAT(1, 16, n_classes, [4, 1], readout, pooling_sizes)
            model = model.to(device)
            train(
                model=model,
                data_loader=data_loader,
                valid_loader=valid_loader,
                epoches=epoches,
                device=device,
                gnn_model=gnn_model,
                readout=readout,
                time=time,
                cv=cv,
                dataset=dataset
            )
            model.load_state_dict(torch.load(f'./checkpoints/{gnn_model}_checkpoint.pt'))
            acc = eval(model, test_loader, device)
            scores.append(acc)
            print(scores)
    np.save(f'./accuracy/{gnn_model}/{dataset}_{readout}_10cv', np.array(scores))
