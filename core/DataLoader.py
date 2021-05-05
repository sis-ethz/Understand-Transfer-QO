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

def filter_out_disabled_decisions(ds, cost_features, decision_feature, disabled_decisions):
    
    for feat in disabled_decisions:
        assert feat in cost_features, f"Feature '{feat}' not in all cost_features. Pls make sure disabled features are in cost features"

    decisions = []

    ds[disabled_decisions] = np.inf
    for i in range(ds.shape[0]):
        # decisions.append()
        # print(ds.iloc[i][cost_features])
        ds.iloc[i, ds.columns.get_loc(decision_feature)] = np.argmin(ds.iloc[i][cost_features].to_numpy())


    return ds


class DataLoader:

    def __init__(self, engine='postgres', base_dir='../../sample_results/'):
        # assert engine in ['postgres',
        #                   'mssql', 'couchbase'], f"Engine {engine} not supported yet!"
        self.engine = engine
        self.base_dir = base_dir

        # self.all_file_locations = {
        #     'tpch': ['customer/{}_orders_customer_optimal.csv', 'orders/{}_customer_orders_optimal.csv',
        #              'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
        #              'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
        #              'partsupp/{}_part_partsupp_optimal.csv', 'partsupp/{}_supplier_partsupp_optimal.csv',
        #              'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv',
        #              'nation/{}_region_nation_optimal.csv', 
        #              'lineitem/{}_orders_lineitem_optimal.csv'
        #              ],
        #     'imdb': ['cast_info/{}_title_cast_info_optimal.csv',
        #              'movie_companies/{}_title_movie_companies_optimal.csv', 'movie_info/{}_title_movie_info_optimal.csv',
        #              'movie_info_idx/{}_title_movie_info_idx_optimal.csv', 'movie_keyword/{}_title_movie_keyword_optimal.csv',
        #              'title/{}_cast_info_title_optimal.csv', 'title/{}_movie_companies_title_optimal.csv',
        #              'title/{}_movie_info_idx_title_optimal.csv', 'title/{}_movie_info_title_optimal.csv',
        #              'title/{}_movie_keyword_title_optimal.csv'
        #              ],
        #     'ssb': ['part/{}_lineorder_part_optimal.csv', 'customer/{}_lineorder_customer_optimal.csv', 'ddate/{}_lineorder_ddate_optimal.csv',
        #             'supplier/{}_lineorder_supplier_optimal.csv'
        #             ]
        # }

        self.all_file_locations = {
            'tpch': ['customer/{}_orders_customer_optimal.csv', 'orders/{}_customer_orders_optimal.csv',
                     'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
                     'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
                     'partsupp/{}_part_partsupp_optimal.csv', 'partsupp/{}_supplier_partsupp_optimal.csv',
                     'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv',
                     'nation/{}_region_nation_optimal.csv', 
                     'lineitem/{}_orders_lineitem_optimal.csv', 'lineitem/{}_part_lineitem_optimal.csv', 
                     'lineitem/{}_partsupp_lineitem_optimal.csv', 'region/{}_nation_region_optimal.csv', 
                     ],
            'tpch_100': ['customer/{}_orders_customer_optimal.csv', 'orders/{}_customer_orders_optimal.csv',
                     'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
                     'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
                     'partsupp/{}_part_partsupp_optimal.csv', 'partsupp/{}_supplier_partsupp_optimal.csv',
                     'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv',
                     'nation/{}_region_nation_optimal.csv', 
                     'lineitem/{}_orders_lineitem_optimal.csv', 'lineitem/{}_part_lineitem_optimal.csv', 
                     'lineitem/{}_partsupp_lineitem_optimal.csv', 'region/{}_nation_region_optimal.csv', 
                     ],
            'tpch_10': ['customer/{}_orders_customer_optimal.csv', 'orders/{}_customer_orders_optimal.csv',
                     'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
                     'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
                     'partsupp/{}_part_partsupp_optimal.csv', 'partsupp/{}_supplier_partsupp_optimal.csv',
                     'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv',
                     'nation/{}_region_nation_optimal.csv', 
                     'lineitem/{}_orders_lineitem_optimal.csv', 'lineitem/{}_part_lineitem_optimal.csv', 
                     'lineitem/{}_partsupp_lineitem_optimal.csv', 'region/{}_nation_region_optimal.csv', 
                     ],
            'tpch_combined': ['customer/{}_orders_customer_optimal.csv', 'orders/{}_customer_orders_optimal.csv',
                     'orders/{}_lineitem_orders_optimal.csv', 'part/{}_lineitem_part_optimal.csv',
                     'part/{}_partsupp_part_optimal.csv', 'partsupp/{}_lineitem_partsupp_optimal.csv',
                     'partsupp/{}_part_partsupp_optimal.csv', 'partsupp/{}_supplier_partsupp_optimal.csv',
                     'supplier/{}_lineitem_supplier_optimal.csv', 'supplier/{}_partsupp_supplier_optimal.csv',
                     'nation/{}_region_nation_optimal.csv', 
                     'lineitem/{}_orders_lineitem_optimal.csv', 'lineitem/{}_part_lineitem_optimal.csv', 
                     'lineitem/{}_partsupp_lineitem_optimal.csv', 'region/{}_nation_region_optimal.csv', 
                     ],
            'imdb': ['cast_info/{}_title_cast_info_optimal.csv',
                     'movie_companies/{}_title_movie_companies_optimal.csv', 'movie_info/{}_title_movie_info_optimal.csv',
                     'movie_info_idx/{}_title_movie_info_idx_optimal.csv', 'movie_keyword/{}_title_movie_keyword_optimal.csv',
                     'title/{}_cast_info_title_optimal.csv', 'title/{}_movie_companies_title_optimal.csv',
                     'title/{}_movie_info_idx_title_optimal.csv', 'title/{}_movie_info_title_optimal.csv',
                     'title/{}_movie_keyword_title_optimal.csv'
                     ],
            'ssb': ['part/{}_lineorder_part_optimal.csv', 'customer/{}_lineorder_customer_optimal.csv', 'ddate/{}_lineorder_ddate_optimal.csv',
                    'supplier/{}_lineorder_supplier_optimal.csv',
                    'lineorder/{}_part_lineorder_optimal.csv', 'lineorder/{}_customer_lineorder_optimal.csv', 
                    'lineorder/{}_ddate_lineorder_optimal.csv', 'lineorder/{}_supplier_lineorder_optimal.csv', 
                    ]
        }

        self.all_ground_truth_files = {
            'tpch': ['part/{}_lineitem_part_optimal_groundtruth.csv',
                     'part/{}_partsupp_part_optimal_groundtruth.csv'],
            'imdb': ['cast_info/{}_title_cast_info_optimal_groundtruth.csv',
                     'movie_companies/{}_title_movie_companies_optimal_groundtruth.csv', 'movie_info/{}_title_movie_info_optimal_groundtruth.csv',
                     'movie_info_idx/{}_title_movie_info_idx_optimal_groundtruth.csv', 'movie_keyword/{}_title_movie_keyword_optimal_groundtruth.csv'],
            'ssb': ['part/{}_lineorder_part_optimal_groundtruth.csv', 'customer/{}_lineorder_customer_optimal_groundtruth.csv',
                    'ddate/{}_lineorder_ddate_optimal_groundtruth.csv', 'supplier/{}_lineorder_supplier_optimal_groundtruth.csv']
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

        # self.all_features = ['left_cardinality_ratio', 'left_cardinality', 'base_cardinality', 'sel_of_join_pred', 'sel_of_pred_on_indexed_attr', 
        #                         'sel_of_pred_on_non_indexed_attr',
        #                      'sel_of_pred_on_indexed_attr_and_join_pred', 'sel_of_pred_on_non_indexed_attr_and_join_pred', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr',
        #                      'total_sel_on_base_table', 'left_ordered', 'base_ordered', 'result_size', 'predicate_op_num_on_indexed_attr', 'predicate_op_num_on_non_indexed_attr']
        
        self.all_features = ['left_cardinality', 'base_cardinality', 'sel_of_pred_on_indexed_attr', 'sel_of_pred_on_non_indexed_attr', 'sel_of_pred_on_indexed_attr_and_join_pred',   
                             'sel_of_pred_on_non_indexed_attr_and_join_pred', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr',
                             'total_sel_on_base_table', 'left_ordered', 'base_ordered', 'left_indexed', 'base_indexed', 'result_size', 'predicate_op_num_on_indexed_attr', 'predicate_op_num_on_non_indexed_attr']

        self.all_feature_idx = [f'feature_{i}' for i in range(len(self.all_features))]

        self.base_features = ['left_cardinality', 'base_cardinality', 'sel_of_pred_on_indexed_attr', 'sel_of_pred_on_non_indexed_attr', 'sel_of_join_pred',
                              'left_ordered', 'base_ordered', 'left_indexed', 'base_indexed', 'result_size', 'predicate_op_num_on_indexed_attr', 
                              'predicate_op_num_on_non_indexed_attr']

        self.aug_features = ['left+right', 'left-right', 'left*right', 'left/right', 'left^2', 'right^2', 'left*logleft', 'right*logright',
                             'sel_of_pred_on_indexed_attr*right', 'sel_of_pred_on_non_indexed_attr*right', 'sel_of_pred_on_indexed_attr_and_join_pred*right', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr*right']

        self.key_features = ['left_cardinality_ratio', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr',
                             'sel_of_pred_on_indexed_attr_and_join_pred', 'sel_of_pred_on_indexed_attr']

        self.viz_features = ['sel_of_pred_on_indexed_attr', 'left_cardinality_ratio']

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
                file_loc = os.path.join(self.base_dir, d, f.format(self.engine))
                if not os.path.exists(file_loc) or not os.path.isfile(file_loc):
                    print(f"[Warning]: file {file_loc} does not exist. Passed.")
                    # dfs.append(pd.DataFrame(columns=dfs[-1].columns))
                else:
                    dfs.append(pd.read_csv(file_loc))
        ds = pd.concat(dfs)
        return self.augment_features(ds)
    
    def get_groundtruth_file_ds(self, return_type='ds', datasets=['ssb', 'tpch', 'imdb'], allow_file_does_not_exist=True):
        gt_dfs = []
        gt_names = []
        for d in datasets:
            for f in self.all_ground_truth_files[d]:
                file_loc = os.path.join(self.base_dir, d, f.format(self.engine))
                if not os.path.exists(file_loc) or not os.path.isfile(file_loc):
                    print(f"[Warning]: file {file_loc} does not exist. Passed.")
                    gt_dfs.append(pd.DataFrame())
                    gt_names.append(' '.join([d, f.format(self.engine)]))
                else:
                    gt_dfs.append(pd.read_csv(file_loc))
                    gt_names.append(' '.join([d, f.format(self.engine)]))

        est_dfs = []
        est_names = []
        for d in datasets:
            for f in self.all_ground_truth_files[d]:
                f = f.replace("_groundtruth", '')
                file_loc = os.path.join(self.base_dir, d, f.format(self.engine))
                if not os.path.exists(file_loc) or not os.path.isfile(file_loc):
                    print(f"[Warning]: file {file_loc} does not exist. Passed.")
                    est_dfs.append(pd.DataFrame())
                    est_names.append(' '.join([d, f.format(self.engine)]))
                else:
                    est_dfs.append(pd.read_csv(file_loc))
                    est_names.append(' '.join([d, f.format(self.engine)]))
        

        for idx, ds in enumerate(gt_dfs):
            if gt_dfs[idx].shape[0] > 0 and len(gt_dfs[idx].columns) > 0:
                gt_dfs[idx] = self.augment_features(ds)
    
        for idx, ds in enumerate(est_dfs):
            if est_dfs[idx].shape[0] > 0 and len(est_dfs[idx].columns) > 0:
                est_dfs[idx] = self.augment_features(ds)

        if return_type == 'ds and names':
            return gt_dfs, est_dfs, gt_names, est_names
        else:
            return gt_dfs, est_dfs


    def get_one_file_ds(self, return_type='ds', datasets=['ssb', 'tpch', 'imdb'], allow_file_does_not_exist=True, suffix=None):
        dfs = []
        names = []
        for d in datasets:
            for f in self.all_file_locations[d]:
                file_loc = os.path.join(self.base_dir, d, f.format(self.engine))
                if suffix is not None:
                    file_loc = file_loc.replace('.csv', f'_{suffix}.csv')
                if not os.path.exists(file_loc) or not os.path.isfile(file_loc):
                    print(f"[Warning]: file {file_loc} does not exist. Passed.")
                    dfs.append(pd.DataFrame())
                    names.append(' '.join([d, f.format(self.engine)]))
                else:
                    dfs.append(pd.read_csv(file_loc))
                    names.append(' '.join([d, f.format(self.engine)]))

        for idx, ds in enumerate(dfs):
            if dfs[idx].shape[0] > 0 and len(dfs[idx].columns) > 0:
                dfs[idx] = self.augment_features(ds)
        if return_type == 'ds and names':
            return dfs, names
        else:
            return dfs
    
    def combine_different_scalers(self, datasets=['tpch', 'tpch_10', 'tpch_100'], target_dataset='tpch_combined', save_target=True):
        combined_ds = []
        for f in self.all_file_locations[datasets[0]]:
            dfs = []
            for d in datasets:
                file_loc = os.path.join(self.base_dir, d, f.format(self.engine))
                dfs.append(pd.read_csv(file_loc))
            ds = pd.concat(dfs)
            combined_ds.append(ds)
            if save_target:
                combined_file_loc = os.path.join(self.base_dir, target_dataset, f.format(self.engine))
                ds.to_csv(combined_file_loc)
        return combined_ds

    
    def transfer_features(self, datasets=['ssb'], from_engine='postgres', write_back=True, return_type='ds'):
        assert 'couchbase' in self.engine, "Transfer_features only support couchbase"
        dfs = []
        names = []
        transfered_features = ['query_id'] + self.all_features
        for d in datasets:
            for f in self.all_file_locations[d]:
                
                raw_file_loc = os.path.join(self.base_dir, d, f.format(self.engine))

                if not os.path.exists(raw_file_loc) or not os.path.isfile(raw_file_loc):
                    print(f"[Warning]: file {raw_file_loc} does not exist. Passed.")
                else:
                    raw_df = pd.read_csv(raw_file_loc)
                    from_df = pd.read_csv(os.path.join(
                        self.base_dir, d, f.format(self.engine).replace(self.engine, from_engine) ))
                    raw_df[transfered_features] = from_df[transfered_features]
                    dfs.append(raw_df)
                    names.append(' '.join([d, f.format(self.engine)])) 
                    if write_back:
                        raw_df.to_csv(os.path.join(self.base_dir, d, f.format(self.engine)))  
        
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
        df['min_sel_on_attr'] = np.min(df[['sel_of_pred_on_indexed_attr', 'sel_of_pred_on_non_indexed_attr']], axis=1)
        return df
