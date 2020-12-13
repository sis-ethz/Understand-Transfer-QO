from core.models.Regressor import Regressor
from core.models.Classifier import Classifier
from sklearn import svm


class SVMClassifier(Classifier):
    
    def __init__(self,):
        super(SVMClassifier, self).__init__()
    
    def fit(self, X, y, sample_weight=None):
        self.model = svm.SVC()
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class SVMRegressor(Regressor):
    def __init__(self,):
        super(SVMRegressor, self).__init__()

    def fit(self, X, y, y_scaler='exp', sample_weight=None):
        if y_scaler is None:
            rgr = svm.SVR().fit(X, y, sample_weight=sample_weight)
            self.model = rgr
            self.y_inv_scale_func = lambda x: x
            return self
        elif y_scaler == 'exp':
            rgr = Esvm.SVR().fit(X, np.log(y), sample_weight=sample_weight)
            self.y_inv_scale_func = np.exp
            self.model = rgr
            return self
        else:
            assert False, f"y-scaler '{y_scaler}' is not supported"

    def predict(self, X):
        return self.y_inv_scale_func(self.model.predict(X))