from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from core.models.Regressor import Regressor
from core.models.Classifier import Classifier


class m_RandomForestRegressor(Regressor):

    def __init__(self,):
        super(Regressor).__init__()

    def fit(self, X_train, y_train, n_estimator=1, max_depth=10, sample_weight=None, y_scaler='exp'):
        if y_scaler is None:
            rgr = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth).fit(
                X_train, y_train, sample_weight=sample_weight)
            self.model = rgr
            self.y_inv_scale_func = lambda x: x
            return self
        elif y_scaler == 'exp':
            rgr = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth).fit(
                X_train, np.log(y_train), sample_weight=sample_weight)
            self.y_inv_scale_func = np.exp
            self.model = rgr
            return self
        else:
            assert False, f"y-scaler '{y_scaler}' is not supported"

    def predict(self, X):
        return self.y_inv_scale_func(self.model.predict(X))

    def __getitem__(self, key):
        return self.model.estimators_[key]


class m_RandomForestClassifier(Classifier):

    def __init__(self,):
        pass

    def fit(self, X_train, y_train, n_estimator=1, max_depth=10, sample_weight=None):
        clf = RandomForestClassifier(
            n_estimators=n_estimator, max_depth=max_depth).fit(X_train, y_train, sample_weight=sample_weight)
        self.model = clf
        return self

    def predict(self, X):
        return self.model.predict(X)

    def __getitem__(self, key):
        return self.model.estimators_[key]
