import os
import pandas as pd
import numpy as np
from scipy.special import softmax


def calculate_confidence_from_costs(costs_arr, optimal_decision=None):
    """
    Calculation equation:
    0. costs = log(costs)
    1. costs = costs / min_cost:
        [2, 2, 3, 4, 5] => [1, 1, 1.5, 2, 2.5]
    2. costs = 1 / costs
    3. costs = softmax(costs)

    Arg: optimal_decision is for validation
    """

    costs_arr = costs_arr.copy()

    # costs_arr = np.log(costs_arr)
    # costs_arr = costs_arr / np.min(costs_arr, axis=1).reshape(-1, 1)
    costs_arr = 1 / costs_arr
    costs_arr = softmax(costs_arr, axis=1)

    # costs_arr_dup = np.zeros(costs_arr.shape)
    # for i in range(costs_arr.shape[0]):
    #     costs_arr_dup[i, optimal_decision[i]] = 1
    # costs_arr = costs_arr_dup

    # # Start validation
    if optimal_decision is not None:
        labels = np.argmax(costs_arr, axis=1)
        assert np.sum(labels - optimal_decision) == 0
    return costs_arr

def calculate_importance_from_costs(costs_arr):

    costs_arr = costs_arr.copy()

    arg_min = np.argmin(costs_arr, axis=1)
    min_costs = np.min(costs_arr, axis=1)
    for i in range(costs_arr.shape[0]):
        costs_arr[i, arg_min[i]] = np.inf
    second_min_costs = np.min(costs_arr, axis=1)
    return (second_min_costs - min_costs) / min_costs

class DataLoader:

    def __init__(self, engine='postgres', base_dir='../../sample_results/'):
        assert engine in ['postgres',
                          'mssql'], f"Engine {engine} not supported yet!"
        self.engine = engine
        self.base_dir = base_dir

        self.all_file_locations = {
            'tpch': ['customer/{}_orders_customer_optimal.csv', 'orders/{}_customer_orders_optimal.csv',
                     'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
                     'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
                     'partsupp/{}_part_partsupp_optimal.csv', 'partsupp/{}_supplier_partsupp_optimal.csv',
                     'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv'],
            'imdb': ['cast_info/{}_title_cast_info_optimal.csv',
                     'movie_companies/{}_title_movie_companies_optimal.csv', 'movie_info/{}_title_movie_info_optimal.csv',
                     'movie_info_idx/{}_title_movie_info_idx_optimal.csv', 'movie_keyword/{}_title_movie_keyword_optimal.csv',
                     'title/{}_cast_info_title_optimal.csv', 'title/{}_movie_companies_title_optimal.csv',
                     'title/{}_movie_info_idx_title_optimal.csv', 'title/{}_movie_info_title_optimal.csv',
                     'title/{}_movie_keyword_title_optimal.csv'
                     ],
            'ssb': ['part/{}_lineorder_part_optimal.csv', 'customer/{}_lineorder_customer_optimal.csv', 'ddate/{}_lineorder_ddate_optimal.csv',
                    'supplier/{}_lineorder_supplier_optimal.csv'
                    ]
        }

        self.clustered_file_locations = file_locations = {
            'tpch': ['customer/{}_orders_customer_optimal.csv',
                     'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
                     'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
                     'partsupp/{}_part_partsupp_optimal.csv',
                     'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv'],
            'imdb': ['title/{}_cast_info_title_optimal.csv', 'title/{}_movie_companies_title_optimal.csv',
                     'title/{}_movie_info_idx_title_optimal.csv', 'title/{}_movie_info_title_optimal.csv',
                     'title/{}_movie_keyword_title_optimal.csv'
                     ],
            'ssb': ['part/{}_lineorder_part_optimal.csv', 'customer/{}_lineorder_customer_optimal.csv', 'supplier/{}_lineorder_supplier_optimal.csv'
                    ]
        }

        self.one_file_location = 'ssb/part/{}_lineorder_part_optimal.csv'

        self.all_features = ['left_cardinality', 'base_cardinality', 'sel_of_join_pred', 'sel_of_pred_on_indexed_attr', 'sel_of_pred_on_non_indexed_attr',
                             'sel_of_pred_on_indexed_attr_and_join_pred', 'sel_of_pred_on_non_indexed_attr_and_join_pred', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr',
                             'total_sel_on_base_table', 'left_ordered', 'base_ordered', 'result_size', 'predicate_op_num_on_indexed_attr', 'predicate_op_num_on_non_indexed_attr']

        self.aug_features = ['left_cardinality_ratio', 'left+right', 'left-right', 'left*right', 'left/right', 'left^2', 'right^2', 'left*logleft', 'right*logright',
                             'sel_of_pred_on_indexed_attr*right', 'sel_of_pred_on_non_indexed_attr*right', 'sel_of_pred_on_indexed_attr_and_join_pred*right', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr*right']

        self.key_features = ['left_cardinality_ratio', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr',
                             'sel_of_pred_on_indexed_attr_and_join_pred', 'sel_of_pred_on_indexed_attr']

        self.viz_features = ['sel_of_pred_on_indexed_attr', 'left_cardinality_ratio',
                             ]

        self.regression_targets = ['hj_idx_cost', 'hj_seq_cost',
                                   'nl_idx_cost', 'nl_seq_cost',  'mj_idx_cost', 'mj_seq_cost']

        self.classification_target = ['optimal_decision']

        self.piecewise_split_features = ['left_ordered', 'base_ordered']

        self.costs = self.regression_targets

    def get_clustered_files_ds(self, datasets=['ssb', 'tpch', 'imdb']):
        dfs = []

        for d in datasets:
            for f in self.clustered_file_locations[d]:
                dfs.append(pd.read_csv(os.path.join(
                    self.base_dir, d, f.format(self.engine))))
        ds = pd.concat(dfs)
        return self.augment_features(ds)

    def get_all_files_ds(self, datasets=['ssb', 'tpch', 'imdb']):
        dfs = []
        for d in datasets:
            for f in self.all_file_locations[d]:
                dfs.append(pd.read_csv(os.path.join(
                    self.base_dir, d, f.format(self.engine))))
        ds = pd.concat(dfs)
        return self.augment_features(ds)

    def get_one_file_ds(self, return_type='ds'):
        # ds = pd.read_csv(os.path.join(
        #     "../../sample_results/", self.one_file_location.format(self.engine)))
        # return self.augment_features(ds)
        dfs = []
        names = []
        for d in ['ssb', 'tpch', 'imdb']:
            for f in self.all_file_locations[d]:
                dfs.append(pd.read_csv(os.path.join(
                    self.base_dir, d, f.format(self.engine))))
                names.append(' '.join([d, f.format(self.engine)]))

        for idx, ds in enumerate(dfs):
            dfs[idx] = self.augment_features(ds)
        if return_type == 'ds and names':
            return dfs, names
        else:
            return dfs

    def augment_features(self, df,):
        df['left+right'] = df['left_cardinality'] + \
            df['base_cardinality']
        df['left*right'] = df['left_cardinality'] * \
            df['base_cardinality']
        df['left/right'] = df['left_cardinality'] / \
            df['base_cardinality']
        df['left-right'] = df['left_cardinality'] - \
            df['base_cardinality']
        df['left^2'] = df['left_cardinality'] * \
            df['left_cardinality']
        df['right^2'] = df['base_cardinality'] * \
            df['base_cardinality']
        df['left*logleft'] = df['left_cardinality'] * \
            np.log(df['left_cardinality'])
        df['right*logright'] = df['base_cardinality'] * \
            np.log(df['base_cardinality'])
        df['sel_of_pred_on_indexed_attr*right'] = df['sel_of_pred_on_indexed_attr'] * \
            df['base_cardinality']
        df['sel_of_pred_on_non_indexed_attr*right'] = df['sel_of_pred_on_non_indexed_attr'] * \
            df['base_cardinality']
        df['sel_of_pred_on_indexed_attr_and_join_pred*right'] = df['sel_of_pred_on_indexed_attr_and_join_pred'] * \
            df['base_cardinality']
        df['sel_of_pred_on_indexed_attr_and_non_indexed_attr*right'] = df['sel_of_pred_on_indexed_attr_and_non_indexed_attr'] * \
            df['base_cardinality']
        return df
