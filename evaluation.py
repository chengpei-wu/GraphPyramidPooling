import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from utils import load_dgl_data


def evaluation(dataset, pooling_sizes, classifier, params=None, fold=10, times=10):
    # Default execution 10 times of 10-fold cross validation.
    scaler = MinMaxScaler()
    x, y = load_dgl_data(
        dataset=dataset,
        pooling_sizes=pooling_sizes,
        rank_label='degree',
        pooling_attr=['degree'],
        pooling_way='mean'
    )
    x = scaler.fit_transform(x)
    scores = []
    for i in range(times):
        kf = StratifiedKFold(n_splits=fold, shuffle=True)
        if classifier == 'SVM':
            svm_rbf = SVC(
                kernel='rbf'
            )
            if params:
                svm_rbf.set_params(**params)
            cv_score = cross_val_score(svm_rbf, x, np.argmax(y, axis=1), cv=kf)
        if classifier == 'RF':
            rfc = RandomForestClassifier(oob_score=True)
            cv_score = cross_val_score(rfc, x, np.argmax(y, axis=1), cv=kf)
        if classifier == 'GBDT':
            gbdt = GradientBoostingClassifier()
            cv_score = cross_val_score(gbdt, x, np.argmax(y, axis=1), cv=kf)
        scores.append(cv_score)
    print(f'\nFinished {times} times of {fold}-fold cross validation.\nACCURACY SCORE: {np.mean(scores)}')
