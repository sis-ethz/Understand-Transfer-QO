from core.models.Regressor import Regressor
from core.models.Classifier import Classifier
from core.models.GAM import *

import numpy as np
import pandas as pd

class piecewiseLinearClassifier(Classifier):
    def __init__(self, model_type=LinearGAMClassifier):
        self.model_type = model_type

    def split_Xs(self, X,  split_X_features, X_conditions, y=None, sample_weight=None):
        self.conditions = X_conditions

        Xs = [np.zeros([1, X.shape[-1]]) for i in range(len(X_conditions))]
        ys = [np.zeros([1]).reshape(1, -1) for i in range(len(X_conditions))]
        sample_weights = [np.zeros([1]).reshape(1, -1) for i in range(len(X_conditions))]

        for i in range(X.shape[0]):
            for idx, cond in enumerate(X_conditions):
                if cond(split_X_features[i]):
                    Xs[idx] = np.concatenate((Xs[idx], X[i, :].reshape(1, -1)), axis=0)
                    if y is not None:
                        ys[idx] = np.concatenate((ys[idx], y[i].reshape(1, -1)), axis=0)
                    if sample_weight is not None:
                        sample_weights[idx] = np.concatenate((sample_weights[idx], sample_weight[i].reshape(1, -1)), axis=0)
                    break

        Xs = [X[1:, :] for X in Xs]
        ys = [y[1:].flatten() for y in ys]
        sample_weights = [sample_weight[1:].flatten() for sample_weight in sample_weights]
        if sample_weight is not None:
            return Xs, ys, sample_weights
        else:
            return Xs, ys, [None for i in range(len(X_conditions))]
    
    def fit(self, X,  X_split_features, X_conditions, y, sample_weight=None):
        self.models = []
        Xs, ys, sample_weights = self.split_Xs(X,  X_split_features, X_conditions, y=y, sample_weight=sample_weight)
        for idx, X, y, sample_weight in zip([i+1 for i in range(len(Xs))], Xs, ys, sample_weights):
            print(f"Sample size for subspace {idx}: {X.shape[0]}")
            model = self.model_type().fit(X, y, sample_weight=sample_weight)
            self.models.append(model)
        return self

    def predict(self, X, condition_idx):
        y = self.models[condition_idx].predict(X)
        return y

    def score(self, X, y, condition_idx):
        return self.models[condition_idx].score(X, y)
    
    def score(self, X,  X_split_features, y):
        Xs, ys, _ = self.split_Xs(X,  X_split_features, self.conditions, y=y, sample_weight=None)
        scores = []
        for X, y, model in zip(Xs, ys, self.models):
            if len(y) > 0:
                scores.append(model.score(X, y))
            else:
                scores.append(None)
        return scores


class piecewiseLinearRegressor(Regressor):

    def __init__(self, model_type=LinearGAMRegressor):
        self.model_type = model_type

    def split_Xs(self, X,  split_X_features, X_conditions, y=None, sample_weight=None):
        self.conditions = X_conditions

        Xs = [np.zeros([1, X.shape[-1]]) for i in range(len(X_conditions))]
        ys = [np.zeros([1]).reshape(1, -1) for i in range(len(X_conditions))]
        sample_weights = [np.zeros([1]).reshape(1, -1) for i in range(len(X_conditions))]

        for i in range(X.shape[0]):
            for idx, cond in enumerate(X_conditions):
                if cond(split_X_features[i]):
                    Xs[idx] = np.concatenate((Xs[idx], X[i, :].reshape(1, -1)), axis=0)
                    if y is not None:
                        ys[idx] = np.concatenate((ys[idx], y[i].reshape(1, -1)), axis=0)
                    if sample_weight is not None:
                        sample_weights[idx] = np.concatenate((sample_weights[idx], sample_weight[i].reshape(1, -1)), axis=0)
                    break

        Xs = [X[1:, :] for X in Xs]
        ys = [y[1:].flatten() for y in ys]
        sample_weights = [sample_weight[1:].flatten() for sample_weight in sample_weights]
        if sample_weight is not None:
            return Xs, ys, sample_weights
        else:
            return Xs, ys, [None for i in range(len(X_conditions))]
    
    def fit(self, X,  X_split_features, X_conditions, y, sample_weight=None):
        self.models = []
        Xs, ys, sample_weights = self.split_Xs(X,  X_split_features, X_conditions, y=y, sample_weight=sample_weight)
        for X, y, sample_weight in zip(Xs, ys, sample_weights):
            model = self.model_type().fit(X, y, sample_weight=sample_weight)
            self.models.append(model)
        return self

    def predict(self, X, condition_idx):
        y = self.models[condition_idx].predict(X)
        return y

    def score(self, X, y, condition_idx):
        return self.models[condition_idx].score(X, y)
    
    def score(self, X,  X_split_features, y):
        Xs, ys, _ = self.split_Xs(X,  X_split_features, self.conditions, y=y, sample_weight=None)
        scores = []
        for X, y, model in zip(Xs, ys, self.models):
            if len(y) > 0:
                scores.append(model.score(X, y))
            else:
                scores.append(None)
        return scores

