
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from core.models.Regressor import Regressor
from core.models.Classifier import Classifier
import numpy as np


class EBMClassifier(Classifier):

    def __init__(self,):
        super(EBMClassifier, self).__init__()

    def fit(self, X, y, sample_weight=None):
        self.model = ExplainableBoostingClassifier().fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class EBMRegressor(Regressor):

    def __init__(self,):
        super(EBMRegressor, self).__init__()

    def fit(self, X, y, y_scaler='exp', sample_weight=None):
        if y_scaler is None:
            rgr = ExplainableBoostingRegressor().fit(
                X, y)
            self.model = rgr
            self.y_inv_scale_func = lambda x: x
            return self
        elif y_scaler == 'exp':
            rgr = ExplainableBoostingRegressor().fit(
                X, np.log(y))
            self.y_inv_scale_func = np.exp
            self.model = rgr
            return self
        else:
            assert False, f"y-scaler '{y_scaler}' is not supported"

    def predict(self, X):
        return self.y_inv_scale_func(self.model.predict(X))