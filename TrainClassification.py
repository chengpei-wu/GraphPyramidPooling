from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils import load_dgl_data
from parameters import *
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

scaler = MinMaxScaler()
for data_name, pooling_sizes in zip(datasets.keys(), datasets.values()):
    x, y = load_dgl_data(
        dataset=data_name,
        pooling_sizes=pooling_sizes,
        rank_label=rank_label,
        pooling_attr=pooling_attr,
        pooling_way=pooling_way
    )
    x = scaler.fit_transform(x)
    all_accuracy = []
    for i in range(10):
        accuracy = []
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(x):
            # kf = StratifiedKFold(n_splits=10, shuffle=True)
            # for train_index, test_index in kf.split(x, np.argmax(y, axis=1)):
            x_train, y_train = x[train_index], y[train_index]
            x_test, y_test = x[test_index], y[test_index]
            acc = 0
            if classifier == 'SVM':
                svm_rbf = SVC(
                    kernel='rbf',
                )
                grid_param = {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'gamma': [0.01, 0.1, 1, 10, 100]
                }
                grid_search = GridSearchCV(
                    estimator=svm_rbf,
                    param_grid=grid_param,
                    verbose=0
                )
                grid_search.fit(x_train, np.argmax(y_train, axis=1))
                svm_rbf.C = grid_search.best_params_['C']
                svm_rbf.gamma = grid_search.best_params_['gamma']
                print(grid_search.best_params_, grid_search.best_score_)
                svm_rbf.fit(x_train, np.argmax(y_train, axis=1))
                acc = svm_rbf.score(x_test, np.argmax(y_test, axis=1))
            if classifier == 'RF':
                rfc = RandomForestClassifier(oob_score=True)
                rfc.fit(x_train, y_train)
                acc = rfc.score(x_test, y_test)
            if classifier == 'GBDT':
                gbdt = GradientBoostingClassifier(learning_rate=0.01)
                gbdt.fit(x_train, np.argmax(y_train, axis=1))
                acc = gbdt.score(x_test, np.argmax(y_test, axis=1))
            if classifier == 'MLP':
                mlp = MLPClassifier(
                    hidden_layer_sizes=(512, 1024, 1024, 512),
                    solver="adam",
                    alpha=1e-5,
                    max_iter=500,
                )
                mlp.fit(x_train, y_train)
                acc = mlp.score(x_test, y_test)
            accuracy.append(acc)
        print(np.mean(accuracy))
        all_accuracy.append(accuracy)
    all_accuracy = np.array(all_accuracy)
    # print(f'{data_name}(max): ', np.max(np.mean(all_accuracy, axis=1)) * 100)
    print(f'{data_name}(mean): ', np.mean(all_accuracy) * 100)
    print(f'{data_name}(std): ', (np.std(all_accuracy)) * 100)
    np.save(f'./accuracy/{classifier}/{data_name}_{rank_label}_10cv', np.array(all_accuracy))
