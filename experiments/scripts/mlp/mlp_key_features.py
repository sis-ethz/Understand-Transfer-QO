# %%

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

class mlp_classifier(nn.Module):

    def __init__(self, n_layers=5, n_features=7, n_classes=6):
        super(mlp_classifier, self).__init__()
        self.layers = nn.ModuleList()
        self.n_classes = n_classes
        if n_layers >= 1:
            self.layers.append(nn.Linear(n_features, 100))
            self.layers.append(nn.ReLU(100))
            for _ in range(n_layers - 2):
                self.layers.append(nn.Linear(100, 100))
                self.layers.append(nn.ReLU(100))
            self.layers.append(nn.Linear(100, n_classes))
            self.layers.append(nn.ReLU(n_classes))
        else:
            self.layers.append(nn.Linear(n_features, n_classes))
            self.layers.append(nn.ReLU(n_classes))
        # self.layers.append(torch.nn.Softmax(n_classes))

    def forward(self, X):
        y = X
        for m in self.layers:
            y = m(y)
        return y


class mlp_regressor(nn.Module):

    def __init__(self, n_layers=5, n_features=7):
        super(mlp_regressor, self).__init__()
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


class mlp_classifier_model():
    def __init__(self, n_layers=5, n_features=7, n_classes=6):
        self.n_classes = n_classes
        self.m_model = mlp_classifier(n_layers, n_features, n_classes)

    def fit(self, X_train, y_train, batch_size=500, max_iter=100, device='cpu', debug_print=False):
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

        optimizer = torch.optim.Adam(self.m_model.parameters(), lr=0.001)

        self.m_model.to(device)

        for epoch in range(max_iter):
            for step, (batch_x, batch_y) in enumerate(loader):

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).long()

                pred_y = self.m_model.forward(batch_x)
                # print(pred_y)
                # exit(1)
                loss = criterion(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if debug_print and step % 10 == 0:
                    print('epoch {}, step {}, train loss {}'.format(
                        epoch, step, loss.data))
        return self

    def predict(self, X):
        self.m_model.to('cpu')
        self.m_model.eval()
        pred_y = self.m_model.forward(torch.Tensor(X))
        _, y_pred_tags = torch.max(pred_y, dim=1)
        return np.array(y_pred_tags)


class mlp_regressor_model():
    def __init__(self, n_layers=5, n_features=7):
        self.m_model = mlp_regressor(n_layers, n_features)

    def fit(self, X_train, y_train, batch_size=200, max_iter=100, device='cpu', debug_print=True):
        # transform y into list of digits
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1)

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
            self.m_model.parameters(), lr=0.001, weight_decay=0.0001)

        self.m_model.to(device)

        for epoch in range(max_iter):
            for step, (batch_x, batch_y) in enumerate(loader):

                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()

                pred_y = self.m_model.forward(batch_x)
                loss = criterion(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if debug_print and step % 10 == 0:
                    print('epoch {}, step {}, train loss {}'.format(
                        epoch, step, loss.data))
        return self

    def predict(self, X):
        self.m_model.to('cpu')
        self.m_model.eval()
        pred_y = self.m_model.forward(torch.Tensor(X))
        return pred_y.detach().numpy()


# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%

file_locations = {
    'tpch': ['customer/orders_customer_optimal_rep.csv', 'orders/customer_orders_optimal_rep.csv',
             'orders/lineitem_orders_optimal_rep.csv', 'part/lineitem_part_optimal_rep.csv',
             'part/partsupp_part_optimal_rep.csv', 'partsupp/lineitem_partsupp_optimal_rep.csv',
             'partsupp/part_partsupp_optimal_rep.csv', 'partsupp/supplier_partsupp_optimal_rep.csv',
             'supplier/lineitem_supplier_optimal_rep.csv', 'supplier/partsupp_supplier_optimal_rep.csv'],
    'imdb': ['cast_info/title_cast_info_optimal_rep.csv',
             'movie_companies/title_movie_companies_optimal_rep.csv', 'movie_info/title_movie_info_optimal_rep.csv',
             'movie_info_idx/title_movie_info_idx_optimal_rep.csv', 'movie_keyword/title_movie_keyword_optimal_rep.csv',
             'title/cast_info_title_optimal_rep.csv', 'title/movie_companies_title_optimal_rep.csv',
             'title/movie_info_idx_title_optimal_rep.csv', 'title/movie_info_title_optimal_rep.csv',
             'title/movie_keyword_title_optimal_rep.csv'
             ],
    'ssb': ['customer/lineorder_customer_optimal_rep.csv', 'ddate/lineorder_ddate_optimal_rep.csv',
            'part/lineorder_part_optimal_rep.csv', 'supplier/lineorder_supplier_optimal_rep.csv'
            ]
}


datasets = ['ssb', 'tpch', 'imdb']


dfs = []


for d in datasets:
    for f in file_locations[d]:
        dfs.append(pd.read_csv(os.path.join("../../../data/", d, f)))

ds = pd.concat(dfs)

#%% Feature augmentation

ds['left+right'] = ds['left_cardinality'] + ds['base_cardinality']
ds['left*right'] = ds['left_cardinality'] * ds['base_cardinality']
ds['left/right'] = ds['left_cardinality'] / ds['base_cardinality']
ds['left-right'] = ds['left_cardinality'] - ds['base_cardinality']
ds['left^2'] = ds['left_cardinality'] * ds['left_cardinality']
ds['right^2'] = ds['base_cardinality'] * ds['base_cardinality']

# %%
m_regression_model = mlp_regressor_model
m_classification_model = mlp_classifier_model

# %% [markdown]
# # Classification Task

#%%
all_features = ['left_cardinality', 'base_cardinality',
       'selectivity_on_indexed_attr', 'left_ordered', 'base_ordered',
       'result_size', 'sel_on_indexed_attr_with_join_predicate']

augmented_features = all_features + \
    ['left+right', 'left*right', 'left/right',
        'left-right', 'left-right', 'left^2', 'right^2']

key_features = ['left_cardinality', 'base_cardinality', 'selectivity_on_indexed_attr']#,
       #'result_size']

features = key_features

regression_targets = ['hj_idx_cost', 'hj_seq_cost', 'nl_idx_cost', 'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost']
classification_target = ['optimal_decision']


X = ds[features]
y = ds['optimal_decision']

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# #%%
clf = m_classification_model(n_features=len(features)).fit(X_train, y_train, max_iter=200,
                                   debug_print=False, device='cuda')
acc = np.sum(clf.predict(X_test) == y_test) / len(y_test)
print(f"Classification test accuracy: %.2f%%" % (acc * 100))
# %% [markdown]
# # Regression Task

# %%
# Collect all the regressors
regressors = {}


all_features = ['left_cardinality', 'base_cardinality',
                'selectivity_on_indexed_attr', 'left_ordered', 'base_ordered',
                'result_size', 'sel_on_indexed_attr_with_join_predicate']

key_features = ['left_cardinality', 'base_cardinality',
                'result_size']

features = key_features

regression_targets = ['hj_idx_cost', 'hj_seq_cost',
                      'nl_idx_cost', 'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost']
classification_target = ['optimal_decision']


X = ds[features]
y = ds[regression_targets + classification_target]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

# %% [markdown]
# ### Hash join + index scan

# %%
c_y_train = np.log(y_train['hj_idx_cost'].to_numpy().reshape(-1, 1))
c_y_test = y_test['hj_idx_cost'].to_numpy().reshape(-1, 1)
# y_scaler = preprocessing.MinMaxScaler().fit(c_y_train)
# c_y_train = y_scaler.transform(c_y_train)
# c_y_test = y_scaler.transform(c_y_test)

rgr = m_regression_model(n_features=len(features)).fit(X_train, c_y_train, max_iter=200, device='cuda', debug_print=False)
regressors['hj_idx_cost'] = rgr

loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)

print(f"hj_idx_cost loss in percentage: +- {loss_in_percentage * 100}%")

# %% [markdown]
# ### Hash join + seq scan

# %%
c_y_train = np.log(y_train['hj_seq_cost'].to_numpy().reshape(-1, 1))
c_y_test = y_test['hj_seq_cost'].to_numpy().reshape(-1, 1)

rgr = m_regression_model(n_features=len(features)).fit(
    X_train, c_y_train, max_iter=200, device='cuda', debug_print=False)
regressors['hj_seq_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"hj_seq_cost loss in percentage: +- {loss_in_percentage * 100}%")


# %%
c_y_train = np.log(y_train['nl_idx_cost'].to_numpy().reshape(-1, 1))
c_y_test = y_test['nl_idx_cost'].to_numpy().reshape(-1, 1)

rgr = m_regression_model(n_features=len(features)).fit(
    X_train, c_y_train, max_iter=200, device='cuda', debug_print=False)
regressors['nl_idx_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"nl_idx_cost loss in percentage: +- {loss_in_percentage * 100}%")

# %% [markdown]
# ### nested loop + seq scan

# %%
c_y_train = np.log(y_train['nl_seq_cost'].to_numpy().reshape(-1, 1))
c_y_test = y_test['nl_seq_cost'].to_numpy().reshape(-1, 1)

rgr = m_regression_model(n_features=len(features)).fit(
    X_train, c_y_train, max_iter=200, device='cuda', debug_print=False)
regressors['nl_seq_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"nl_seq_cost loss in percentage: +- {loss_in_percentage * 100}%")
# %% [markdown]
# ### merge join + index scan

# %%
c_y_train = np.log(y_train['mj_idx_cost'].to_numpy().reshape(-1, 1))
c_y_test = y_test['mj_idx_cost'].to_numpy().reshape(-1, 1)

rgr = m_regression_model(n_features=len(features)).fit(
    X_train, c_y_train, max_iter=200, device='cuda', debug_print=False)
regressors['mj_idx_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"mj_idx_cost loss in percentage: +- {loss_in_percentage * 100}%")

# %% [markdown]
# ### merge join + seq scan

# %%
c_y_train = np.log(y_train['mj_seq_cost'].to_numpy().reshape(-1, 1))
c_y_test = y_test['mj_seq_cost'].to_numpy().reshape(-1, 1)

rgr = m_regression_model(n_features=len(features)).fit(X_train, c_y_train, max_iter=200, device='cuda', debug_print=False)
regressors['mj_seq_cost'] = rgr

loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"mj_seq_cost loss in percentage: +- {loss_in_percentage * 100}%")
# %% [markdown]
# # Use Regression model to do classification

# %%
predict_test = []
operators = ['hj_idx_cost', 'hj_seq_cost', 'nl_idx_cost',
             'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost']

for op_idx, op in enumerate(operators):
    predict_test.append(regressors[op].predict(X_test))

results = np.stack(predict_test, axis=1)
acc = np.sum(np.argmin(results, axis=1).reshape(-1, 1).flatten() ==
             y_test['optimal_decision'].to_numpy().flatten()) / len(y_test)
print("Test accuracy using all learned costs: %.2f%%" % (acc*100))


# %%
