# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np
import os
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', dest='max_depth', type=int, default=10)
    parser.add_argument(
        '--n_estimators', dest='n_estimators', type=int, default=10)
    return parser


parser = arg_parser()
args = parser.parse_args()
max_depth = args.max_depth
n_estimators = args.n_estimators

print("============================================")
print(f"Using max_depth = {max_depth}, n_estimators = {n_estimators}")

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

# %% Feature augmentation

ds['left+right'] = ds['left_cardinality'] + ds['base_cardinality']
ds['left*right'] = ds['left_cardinality'] * ds['base_cardinality']
ds['left/right'] = ds['left_cardinality'] / ds['base_cardinality']
ds['left-right'] = ds['left_cardinality'] - ds['base_cardinality']
ds['left^2'] = ds['left_cardinality'] * ds['left_cardinality']
ds['right^2'] = ds['base_cardinality'] * ds['base_cardinality']

# %%
m_regression_model = RandomForestRegressor
m_classification_model = RandomForestClassifier

# %% [markdown]
# # Classification Task

# %%
all_features = ['left_cardinality', 'base_cardinality',
                'selectivity_on_indexed_attr', 'left_ordered', 'base_ordered',
                'result_size', 'sel_on_indexed_attr_with_join_predicate']

key_features = ['left_cardinality', 'base_cardinality',
                'result_size']

augmented_features = all_features + \
    ['left+right', 'left*right', 'left/right',
        'left-right', 'left-right', 'left^2', 'right^2']

features = all_features

regression_targets = ['hj_idx_cost', 'hj_seq_cost',
                      'nl_idx_cost', 'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost']
classification_target = ['optimal_decision']


X = ds[features]
y = ds['optimal_decision']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, stratify=y, random_state=1)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)


# %%
clf = m_classification_model(
    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=2).fit(X_train, y_train)

acc = np.sum(clf.predict(X_test) == y_test) / len(y_test)
print(f"Test accuracy: %.2f%%" % (acc * 100))

# %% [markdown]
# # Regression Task

# %%
all_features = ['left_cardinality', 'base_cardinality',
                'selectivity_on_indexed_attr', 'left_ordered', 'base_ordered',
                'result_size', 'sel_on_indexed_attr_with_join_predicate']

key_features = ['left_cardinality', 'base_cardinality',
                'result_size']


features = all_features

regression_targets = ['hj_idx_cost', 'hj_seq_cost',
                      'nl_idx_cost', 'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost']
classification_target = ['optimal_decision']


X = ds[features]
y = ds[regression_targets + classification_target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=1)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# %%
# Collect all the regressors
regressors = {}
feature_importances = np.zeros(len(features))
# %% [markdown]
# ### Hash join + index scan

# %%
c_y_train = np.log(y_train['hj_idx_cost'])
c_y_test = y_test['hj_idx_cost']

rgr = m_regression_model(n_estimators=n_estimators,
                         max_depth=max_depth).fit(X_train, c_y_train)
regressors['hj_idx_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"hj_idx_cost loss in percentage: +- {loss_in_percentage * 100}%")

# %% [markdown]
# ### Hash join + seq scan

# %%
c_y_train = np.log(y_train['hj_seq_cost'])
c_y_test = y_test['hj_seq_cost']

rgr = m_regression_model(n_estimators=n_estimators,
                         max_depth=max_depth).fit(X_train, c_y_train)
regressors['hj_seq_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"hj_seq_cost loss in percentage: +- {loss_in_percentage * 100}%")


# %% [markdown]
# Nested loop + idx scan

# %%
c_y_train = np.log(y_train['nl_idx_cost'])
c_y_test = y_test['nl_idx_cost']

rgr = m_regression_model(n_estimators=n_estimators,
                         max_depth=max_depth).fit(X_train, c_y_train)
regressors['nl_idx_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"nl_idx_cost loss in percentage: +- {loss_in_percentage * 100}%")

# %% [markdown]
# ### nested loop + seq scan

# %%
c_y_train = np.log(y_train['nl_seq_cost'])
c_y_test = y_test['nl_seq_cost']

rgr = m_regression_model(n_estimators=n_estimators,
                         max_depth=max_depth).fit(X_train, c_y_train)
regressors['nl_seq_cost'] = rgr
loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"nl_seq_cost loss in percentage: +- {loss_in_percentage * 100}%")

# %% [markdown]
# ### merge join + index scan

# %%
c_y_train = np.log(y_train['mj_idx_cost'])
c_y_test = y_test['mj_idx_cost']

rgr = m_regression_model(n_estimators=n_estimators,
                         max_depth=max_depth).fit(X_train, c_y_train)
regressors['mj_idx_cost'] = rgr

loss_in_percentage = np.average(
    np.abs(np.exp(rgr.predict(X_test)) - c_y_test) / c_y_test)
print(f"mj_idx_cost loss in percentage: +- {loss_in_percentage * 100}%")


# %% [markdown]
# ### merge join + seq scan

# %%
c_y_train = np.log(y_train['mj_seq_cost'])
c_y_test = y_test['mj_seq_cost']

rgr = m_regression_model(n_estimators=n_estimators,
                         max_depth=max_depth).fit(X_train, c_y_train)
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
print("============================================")
