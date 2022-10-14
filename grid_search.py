import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from parameters import *
from utils import load_dgl_data

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
        grid_search.fit(x, np.argmax(y, axis=1))
        print(grid_search.best_params_, grid_search.best_score_)
    if classifier == 'RF':
        rfc = RandomForestClassifier(oob_score=True)
        grid_param = {
            'min_samples_leaf': [1, 2, 5],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [25, 50, 100, 150, 200]
        }
        grid_search = GridSearchCV(
            estimator=rfc,
            param_grid=grid_param,
            verbose=0
        )
        grid_search.fit(x, y)
        print(grid_search.best_params_, grid_search.best_score_)
    if classifier == 'GBDT':
        gbdt = GradientBoostingClassifier()
        grid_param = {
            'min_samples_leaf': [1, 2, 5],
            'min_samples_split': [5, 10, 15],
            'learning_rate': [0.2, 0.1, 0.08],
            'n_estimators': [50, 80, 100, 120]
        }
        grid_search = GridSearchCV(
            estimator=gbdt,
            param_grid=grid_param,
            verbose=0
        )
        grid_search.fit(x, np.argmax(y, axis=1))
        print(grid_search.best_params_, grid_search.best_score_)
