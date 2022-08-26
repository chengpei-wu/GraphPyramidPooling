import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import dgl
import numpy as np
from sklearn.metrics import accuracy_score
from GCN import Classifier, collate

# 创建训练集和测试集
data = dgl.data.TUDataset('MUTAG')
data = [data[id] for id in range(len(data))]
data = shuffle(data)
N = len(data) // 10
acc = []
for K in range(10):
    data_test = data[K * N: (K + 1) * N]
    if K == 0:
        data_train = data[N:]
    else:
        data_train = np.concatenate((data[:K * N], data[(K + 1) * N:]))

    data_loader = DataLoader(data_train, batch_size=32, shuffle=True,
                             collate_fn=collate)

    # 构造模型
    model = Classifier(1, 5, 2, 'nppr', [1, 2, 4, 8, 16])
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
    test_loader = DataLoader(data_test, batch_size=64, shuffle=False,
                             collate_fn=collate)
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
