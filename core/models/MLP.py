from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.utils.data as Data
from tqdm import tqdm

from core.models.Regressor import Regressor
from core.models.Classifier import Classifier

torch.manual_seed(0)


class mlp_classifier_module(nn.Module):

    def __init__(self, n_layers=5, n_features=7, n_classes=6):
        super(mlp_classifier_module, self).__init__()
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        if n_layers >= 1:
            self.layers.append(nn.Linear(n_features, 100))
            self.layers.append(nn.ReLU(100))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(100, 100))
                self.layers.append(nn.ReLU(100))
            self.layers.append(nn.Linear(100, n_classes))
            # self.layers.append(nn.ReLU(n_classes))
        else:
            self.layers.append(nn.Linear(n_features, n_classes))
            self.layers.append(nn.ReLU(n_classes))
        # self.layers.append(nn.Softmax(dim=1))

    def forward(self, X):
        y = X
        for m in self.layers:
            y = m(y)
        return y


class mlp_regressor_module(nn.Module):

    def __init__(self, n_layers=5, n_features=7):
        super(mlp_regressor_module, self).__init__()
        self.layers = nn.ModuleList()
        if n_layers >= 1:
            self.layers.append(nn.Linear(n_features, 100))
            self.layers.append(nn.ReLU())
            for _ in range(n_layers - 1):
                self.layers.append(nn.Linear(100, 100))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(100, 1))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Sigmoid())
            # self.layers.append(nn.Tanh())
        else:
            self.layers.append(nn.Linear(n_features, 1))
            self.layers.append(nn.ReLU())
            # self.layers.append(nn.Sigmoid())

    def forward(self, X):
        y = X
        for m in self.layers:
            y = m(y)
        return y


class MLPClassifier(Classifier):
    def __init__(self, n_layers=5, n_features=2, n_classes=6):
        super(MLPClassifier, self).__init__()
        self.n_classes = n_classes
        self.model = mlp_classifier_module(n_layers, n_features, n_classes)
        self.n_features = n_features

    def fit(self, X_train, y_train, batch_size=500, max_iter=500, device='cpu', debug_print=False, test_kit=None, lr=0.01):
        
        if X_train.shape[-1] != self.n_features:
            self.n_features = X_train.shape[-1]
            self.model = mlp_regressor_module(n_layers, X_train.shape[-1])
        
        # transform y into list of digits
        
        y_train = np.array(y_train)

        X_tensor = torch.Tensor(X_train).to(device)
        y_tensor = torch.Tensor(y_train).to(device)
        # form data loader
        torch_dataset = Data.TensorDataset(
            X_tensor, y_tensor)
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=batch_size,      # mini batch size
            shuffle=True,               #
            # num_workers=5,  #
            # pin_memory=True
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.to(device)

        self.history_acc_during_training = []

        for epoch in tqdm(range(max_iter)):
            for step, (batch_x, batch_y) in enumerate(loader):

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).long()

                pred_y = self.model.forward(batch_x)
                loss = criterion(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                if debug_print and step % 20 == 0:
                    print('epoch {}, step {}, train loss {}'.format(
                        epoch, step, loss.data))
            if test_kit and epoch % 5 == 0:
                acc = self.accuracy(test_kit[0], test_kit[1], device=device)
                self.history_acc_during_training.append([epoch, acc])
        return self

    def predict(self, X, device='cpu'):
        X = torch.Tensor(X).to(device)
        self.model.to(device)
        pred_y = self.model.forward(X)
        pred_y = pred_y.to('cpu')
        _, y_pred_tags = torch.max(pred_y, dim=1)
        return np.array(y_pred_tags)

    def accuracy(self, X, y, device='cpu'):
        return self.score(X, y, device)

    def score(self, X, y, device='cpu'):
        return np.sum(self.predict(X, device=device) == y) / len(y)


class MLPClassifier_with_confidence(MLPClassifier):

    def __init__(self, n_layers=5, n_features=2, n_classes=6):
        super(MLPClassifier_with_confidence, self).__init__(
            n_layers=n_layers, n_features=n_features, n_classes=n_classes)
        self.n_features = n_features

    def fit(self, X_train, y_train, sample_weight=None, batch_size=500, max_iter=500, device='cpu', debug_print=False, test_kit=None, loss_func=None, lr=0.01):

        def manual_cross_entropy(input, target, size_average=True):
            """ 
            Cross entropy that accepts soft targets
            Args:
                pred: predictions for neural network
                targets: targets, can be soft
                size_average: if false, sum is returned instead of mean
            Examples::
                input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
                input = torch.autograd.Variable(out, requires_grad=True)

                target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
                target = torch.autograd.Variable(y1)
                loss = cross_entropy(input, target)
                loss.backward()
            """
            logsoftmax = nn.LogSoftmax(dim=1)
            # print(torch.sum(-target * logsoftmax(input), dim=1))
            if size_average:
                loss = torch.mean(
                    torch.sum(-target * logsoftmax(input), dim=1))
            else:
                loss = torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
            return loss

        if X_train.shape[-1] != self.n_features:
            self.n_features = X_train.shape[-1]
            self.model = mlp_regressor_module(n_layers, X_train.shape[-1])

        # transform y into list of digits
        y_train = np.array(y_train)

        X_tensor = torch.Tensor(X_train).to(device)
        y_tensor = torch.Tensor(y_train).to(device)
        # form data loader
        torch_dataset = Data.TensorDataset(
            X_tensor, y_tensor)
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=batch_size,      # mini batch size
            shuffle=True,               #
            # num_workers=5,
            # pin_memory=True
        )

        criterion = manual_cross_entropy

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.to(device)

        self.history_acc_during_training = []

        for epoch in tqdm(range(max_iter)):
            for step, (batch_x, batch_y) in enumerate(loader):

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred_y = self.model.forward(batch_x)
                loss = criterion(pred_y, batch_y)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                if debug_print and step % 20 == 0:
                    print('epoch {}, step {}, train loss {}'.format(
                        epoch, step, loss.data))
            if test_kit and epoch % 5 == 0:
                acc = self.accuracy(test_kit[0], test_kit[1], device=device)
                self.history_acc_during_training.append([epoch, acc])
        return self


class MLPRegressor(Regressor):
    def __init__(self, n_layers=5, n_features=2):
        super(MLPRegressor, self).__init__()
        self.n_features = n_features
        self.model = mlp_regressor_module(n_layers, n_features)

    def fit(self, X_train, y_train, batch_size=500, max_iter=500, device='cpu', y_scaler='exp', debug_print=False, test_kit=None):
        # transform y into list of digits

        if X_train.shape[-1] != self.n_features:
            self.n_features = X_train.shape[-1]
            self.model = mlp_regressor_module(n_layers, X_train.shape[-1])


        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)

        if y_scaler is None:
            self.y_inv_scale_func = lambda x: x
        elif y_scaler == 'exp':
            y_train = np.log(y_train)
            self.y_inv_scale_func = np.exp
        else:
            assert False, f"y-scaler '{y_scaler}' is not supported"

        X_tensor = torch.Tensor(X_train).to(device)
        y_tensor = torch.Tensor(y_train).to(device)

        # form data loader
        torch_dataset = Data.TensorDataset(
            X_tensor, y_tensor)
        loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=batch_size,      # mini batch size
            shuffle=True,               #
            # num_workers=5,  #
            # pin_memory=True
        )

        criterion = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=0.0001)

        self.model.to(device)

        self.history_loss_during_training = []

        for epoch in tqdm(range(max_iter)):
            for step, (batch_x, batch_y) in enumerate(loader):

                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                pred_y = self.model.forward(batch_x)
                loss = criterion(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if debug_print and step % 10 == 0:
                    print('epoch {}, step {}, train loss {}'.format(
                        epoch, step, loss.data))
            if test_kit and epoch % 5 == 0:
                loss = self.mean_relative_err(
                    test_kit[0], test_kit[1], device=device)
                self.history_loss_during_training.append([epoch, loss])
#                 print("loss = ", loss)

        return self

    def predict(self, X, device='cpu'):
        self.model.to(device)
        X = torch.Tensor(X).to(device)
        pred_y = self.model.forward(X)
        pred_y = pred_y.to('cpu').detach().numpy()
        return self.y_inv_scale_func(pred_y)
