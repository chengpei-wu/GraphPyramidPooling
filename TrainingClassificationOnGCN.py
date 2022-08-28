import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import accuracy_score
from GCN import Classifier, collate

data = dgl.data.TUDataset('DD')
n_classes = data.num_classes
data = np.array([data[id] for id in range(len(data))], dtype=object)
kf = KFold(n_splits=10, shuffle=True)
acc = []
for train_index, test_index in kf.split(data):
    data_train, data_test = data[train_index], data[test_index]
    data_loader = DataLoader(data_train, batch_size=64, shuffle=True, collate_fn=collate)

    print('xxx')
    # 构造模型
    model = Classifier(1, 5, n_classes, 'nppr', [1, 2, 4, 8, 16])
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 模型训练
    model.train()
    epoch_losses = []
    for epoch in range(100):
        epoch_loss = 0
        for iter, (batchg, label) in enumerate(data_loader):
            prediction = model(batchg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print(f'Epoch: {epoch}, loss {epoch_loss}')
        epoch_losses.append(epoch_loss)

    # 测试
    test_loader = DataLoader(data_test, batch_size=64, shuffle=False, collate_fn=collate)
    model.eval()
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            # batchg, label = batchg.to(DEVICE), label.to(DEVICE)
            pred = torch.softmax(model(batchg), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
    print("Test accuracy: ", accuracy_score(test_label, test_pred))
    acc.append(accuracy_score(test_label, test_pred))
print(acc)
