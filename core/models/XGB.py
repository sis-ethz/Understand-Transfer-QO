from core.models.Regressor import Regressor
from core.models.Classifier import *
from xgboost import XGBClassifier
import xgboost as xgb

# xgb_model=XGBRegressor(n_estimators=100, learning_rate=1, 
#                        objective='reg:squarederror', subsample=1, reg_lambda=0.1)

# xgb_model.fit(X, y, sample_weight=w)

class XGBOOSTClassifier(Classifier):
    
    def __init__(self,):
        super(XGBOOSTClassifier, self).__init__()
    
    def fit(self, X, y, sample_weight=None):
        if y[0] * sum(sample_weight > 0) == sum(y * (sample_weight > 0)):
            self.model = unified_predictor(y[0])
        else:
            self.model = XGBClassifier()
            self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

# class SVMRegressor(Regressor):
#     def __init__(self,):
#         super(SVMRegressor, self).__init__()

#     def fit(self, X, y, y_scaler='exp', sample_weight=None):
#         if y_scaler is None:
#             rgr = svm.SVR().fit(X, y, sample_weight=sample_weight)
#             self.model = rgr
#             self.y_inv_scale_func = lambda x: x
#             return self
#         elif y_scaler == 'exp':
#             rgr = svm.SVR().fit(X, np.log(y), sample_weight=sample_weight)
#             self.y_inv_scale_func = np.exp
#             self.model = rgr
#             return self
#         else:
#             assert False, f"y-scaler '{y_scaler}' is not supported"

#     def predict(self, X):
#         return self.y_inv_scale_func(self.model.predict(X))