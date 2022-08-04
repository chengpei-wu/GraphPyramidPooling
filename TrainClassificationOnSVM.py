from utils import load_data, load_dgl_data
from MLP import MLP
from parameters import *
from sklearn.utils import shuffle
import numpy as np
from sklearn.svm import SVC

for data_name in datasets:
    # data_name = 'DD'
    x, y, num_classes,_ = load_dgl_data(pooling_sizes, dataset=data_name)
    print(x.shape, y.shape)
    # 10 times of 10 fold corss validation
    accuracy = []
    for i in range(1):
        x, y = shuffle(x, y)
        N = len(x) // 10
        for K in range(10):
            x_test, y_test = x[K * N: (K + 1) * N], y[K * N: (K + 1) * N]
            if K == 0:
                x_train, y_train = x[N:], y[N:]
            else:
                x_train, y_train = np.concatenate((x[:K * N], x[(K + 1) * N:])), np.concatenate(
                    (y[:K * N], y[(K + 1) * N:]))
            svm_rbf = SVC(kernel='rbf')
            svm_rbf.fit(x_train,np.argmax(y_train,axis=1))                     
            acc = svm_rbf.score(x_test,np.argmax(y_test,axis=1))
            accuracy.append(acc)
            print('test accuracy: ',acc)
    # accuracy =  np.array(accuracy).reshape(-1, 10)
    # print(f'{data_name}: ',np.max(np.mean(accuracy,axis=1)))
    np.save(f'./accuracy/SVM/{data_name}_cv', np.array(accuracy).reshape(-1, 10))