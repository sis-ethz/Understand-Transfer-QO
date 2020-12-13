from pygam import LinearGAM, s, f, LogisticGAM, l
from pygam.datasets import wage
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from sklearn.linear_model import LogisticRegression, LinearRegression

from core.models.Regressor import Regressor
from core.models.Classifier import Classifier

import numpy as np


class LinearGAMClassifier(Classifier):

    def __init__(self,):
        pass

    def fit(self, X, y, sample_weight=None):
        n_features = X.shape[-1]
        linear_term = l(0)
        for i in range(1, n_features):
            linear_term += l(i)

        self.classes = np.unique(y)

        self.clfs = []
        # print("All classes: ", self.classes)

        total_y = np.zeros(len(y))

        for c in self.classes:
            X_train = X
            y_train = np.zeros(len(y))
            class_1 = y == c
            class_0 = y != c
            y_train[class_1] = 1
            y_train[class_0] = 0

#             clf = LogisticGAM(linear_term).gridsearch(X_train, y_train)
            clf = LogisticRegression(max_iter=50000).fit(X_train, y_train, sample_weight=sample_weight)
            self.clfs.append(clf)
        return self

    def predict(self, X):
        y = np.zeros(X.shape[0])
        preds = []
        for idx, c in enumerate(self.classes):
            preds.append(self.clfs[idx].predict_proba(X)[:, -1].reshape(-1, 1))
            # print(preds[-1].shape)

        all_preds = np.concatenate(preds, axis=1)
        return self.classes[np.argmax(all_preds, axis=1)]

    def score(self, X, y):
        acc = np.sum(self.predict(X) == y) / len(y)
        return acc


class LinearGAMRegressor(Regressor):

    def __init__(self,):
        super(LinearGAMRegressor, self).__init__()

    def fit(self, X_train, y_train, sample_weight=None, y_scaler='exp'):
        if y_scaler is None:
            rgr = LinearRegression().fit(
                X_train, y_train, sample_weight=sample_weight)
            self.model = rgr
            self.y_inv_scale_func = lambda x: x
            return self
        elif y_scaler == 'exp':
            rgr = LinearRegression().fit(
                X_train, np.log(y_train), sample_weight=sample_weight)
            self.y_inv_scale_func = np.exp
            self.model = rgr
            return self
        else:
            assert False, f"y-scaler '{y_scaler}' is not supported"

    def predict(self, X):
        return self.y_inv_scale_func(self.model.predict(X))
