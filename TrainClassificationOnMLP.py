from sklearn.model_selection import StratifiedKFold
from utils import load_dgl_data
from MLP import MLP
from parameters import *
import numpy as np

for data_name in datasets:
    x, y, num_classes, num_node_attr = load_dgl_data(pooling_sizes, dataset=data_name)
    all_accuracy = []
    for i in range(10):
        accuracy = []
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(x, np.argmax(y, axis=1)):
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]
            print(x_train.shape, y_train.shape)
            print(x_test.shape, y_test.shape)
            mlp = MLP(
                epochs=epochs,
                batch_size=batch_size,
                valid_proportion=valid,
                model=None,
                pooling_sizes=pooling_sizes,
                num_classes=num_classes,
                num_node_attr=num_node_attr
            )
            mlp.fit(x_train, y_train)
            pred = mlp.my_predict(x_test)
            acc = sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1)) / len(x_test)
            accuracy.append(acc)
        print(np.mean(accuracy))
        all_accuracy.append(accuracy)
    all_accuracy = np.array(all_accuracy)
    print(f'{data_name}(max): ', np.max(np.mean(all_accuracy, axis=1)) * 100)
    print(f'{data_name}(mean): ', np.mean(all_accuracy) * 100)
    print(f'{data_name}(std): ', (np.std(all_accuracy)) * 100)
    np.save(f'./accuracy/MLP//{data_name}_{rank_label}_10cv', np.array(all_accuracy))
