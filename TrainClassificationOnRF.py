from utils import load_data, load_dgl_data
from MLP import MLP
from parameters import *
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  # 网格搜索

for data_name in datasets:
    # data_name = 'DD'
    x, y, num_classes, _ = load_dgl_data(pooling_sizes, dataset=data_name)

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
            rfc = RandomForestClassifier(oob_score=True)
            
            grid_param = {
                'max_depth': [2, 4, 6, 8, 10, 12],
                'min_samples_leaf': [1, 2, 5],
                'min_samples_split': [2, 5, 10],
                'n_estimators': [25, 50, 100, 150, 200]
            }
            grid_search = GridSearchCV(
                estimator=rfc, 
                param_grid=grid_param,  
                cv=10, 
                verbose=3
            )
            grid_search.fit(x_train, y_train)

            rfc.max_depth = grid_search.best_params_['max_depth']
            rfc.min_samples_leaf = grid_search.best_params_['min_samples_leaf']
            rfc.min_samples_split = grid_search.best_params_['min_samples_split']
            rfc.n_estimators = grid_search.best_params_['n_estimators']
            rfc.fit(x_train, y_train)
            acc = rfc.score(x_test, y_test)
            accuracy.append(acc)
            print(accuracy)
    # accuracy =  np.array(accuracy).reshape(-1, 10)
    # print(f'{data_name}: ',np.max(np.mean(accuracy,axis=1)))
    np.save(f'./accuracy/RF/{data_name}_cv', np.array(accuracy).reshape(-1, 10))
