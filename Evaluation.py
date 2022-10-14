import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    for i in range(10):
        accuracy = []
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        if classifier == 'SVM':
            svm_rbf = SVC()
            cv_score = cross_val_score(svm_rbf, x, np.argmax(y, axis=1), cv=kf)
        if classifier == 'RF':
            rfc = RandomForestClassifier(oob_score=True)
            cv_score = cross_val_score(rfc, x, np.argmax(y, axis=1), cv=kf)
        if classifier == 'GBDT':
            gbdt = GradientBoostingClassifier()
            cv_score = cross_val_score(gbdt, x, np.argmax(y, axis=1), cv=kf)
        print(cv_score.mean())
