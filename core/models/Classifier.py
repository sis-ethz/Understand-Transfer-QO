import pickle
import os
import numpy as np

class Classifier():

    def __init__(self,):
        self.model = None

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X):
        pass

class model_wrapper():

    def __init__(self, model=None, scaler=None, model_name='default_model_name', location='/mnt/interpretable-cost-model/experiments/saved_models/'):       
        self.model = model
        self.scaler = scaler
        self.model_name = model_name
        self.location = location

    def predict(self, X):
        X_trans = self.scaler.transform(X)
        return self.model.predict(X_trans)

    def save_model(self):
        location = self.location
        assert self.model_name is not None and location is not None
        pickle.dump(self, open(os.path.join(location, self.model_name + '.pkl'), 'wb'))
    
    def load_model(self):
        location = self.location
        assert self.model_name is not None and location is not None
        return pickle.load(open(os.path.join(location, self.model_name + '.pkl'), 'rb'))

class unified_predictor:

    def __init__(self, val):
        self.val = val
    
    def predict(self, X):
        return np.array([self.val for _ in range(X.shape[0])])

    def score(self, X, y):
        return np.sum(y == self.val) / len(y)

    
        