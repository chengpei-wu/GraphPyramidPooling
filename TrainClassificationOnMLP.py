from utils import load_data, load_dgl_data
from MLP import MLP
from parameters import *
from sklearn.utils import shuffle
import numpy as np

for data_name in datasets:
    # data_name = 'DD'
    x, y, num_classes, num_node_attr = load_dgl_data(pooling_sizes, dataset=data_name)

    # 10 times of 10 fold corss validation
    accuracy = []
    for i in range(10):
        x, y = shuffle(x, y)
        N = len(x) // 10
        for K in range(10):
            x_test, y_test = x[K * N: (K + 1) * N], y[K * N: (K + 1) * N]
            print(K * N, '--', (K + 1) * N)
            if K == 0:
                x_train, y_train = x[N:], y[N:]
                print(N,'--')
            else:
                x_train, y_train = np.concatenate((x[:K * N], x[(K + 1) * N:])), np.concatenate(
                    (y[:K * N], y[(K + 1) * N:]))
                print(0,'--',K * N,'+',(K + 1) * N,'--')
            
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
            mlp.fit(x_train, y_train, f'./models/{data_name}-{K}')

            pred = mlp.my_predict(x_test)
            acc = sum(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1)) / len(x_test)
            accuracy.append(acc)
            print(acc)
    np.save(f'./accuracy/MLP/{data_name}', np.array(accuracy).reshape(10, 10))
