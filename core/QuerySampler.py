import numpy as np
import pandas as pd
from cardinality_estimation_quality.cardinality_estimation_quality import *
import os
import sys
from Visualizer import *
from tqdm import tqdm


class Database:
    def __init__(self, db_url='host=localhost port=5432 user=postgres dbname={} password=postgres', db_name=None):
        self.db_url = db_url
        self.db_name = db_name
        if db_name:
            self.init_db(db_name)

    def init_db(self, db_name):
        db = self.db_url.format(db_name)
        PG = Postgres(db)
        self.db = PG
        return PG

    def disable_parallel(self):
        self.execute(
            'LOAD \'pg_hint_plan\';SET max_parallel_workers_per_gather=0;SET max_parallel_workers=0;', set_env=True)

    def explain(self, query):
        q = QueryResult(None)
        q.query = query
        q.explain(self.db, execute=False)
        return q

    def execute(self, query, set_env=False):
        return self.db.execute(query, set_env=set_env)


class Table:
    def __init__(self, table_name):
        self.table_name = table_name

    def set_features(self, table_size=None, primary_key=None, key_range=None, index_size=None):
        self.table_size = table_size
        self.primary_key = primary_key
        self.key_range = key_range
        self.index_size = index_size
        # and other features


class QuerySampler:

    def __init__(self):
        self.join_graph = None
        self.tables = None
        self.primary_keys = {}  # table: primary key
        self.join_keys = {}  # (table, table) : (key1, key2)
        self.table_features = {}  # table features
        self.db = None  # the database used by sampler
        self.left_size_ratio_threshold = 0.5

    def sample_for_table(self, primary_table, left_tables=[], sample_size=100):
        results = []
        print(f"Samping for {primary_table}, with left tables: {left_tables}")
        for _ in tqdm(range(sample_size)):
            rand_table = left_tables[np.random.randint(len(left_tables))]
            results.append(self.collect_random_query_2_table_all_op(
                primary_table, rand_table))
            # print(results)
            # exit(0)
        return results

    def collect_random_query_2_table_all_op(self, base_table, left_table):

        # ==============================================
        # load features
        base_size = self.table_features[base_table]['table_size']
        key_range = self.table_features[base_table]['key_range']
        left_table_size = self.table_features[left_table]['table_size']
        left_table_key, base_table_key = self.join_keys[(
            left_table, base_table)]
        base_indexed_key = self.primary_keys[base_table]
        # ==============================================

        if self.left_size_ratio_threshold < 0:
            left_size_ratio_threshold = left_table_size / base_size
        else:
            left_size_ratio_threshold = min(
                self.left_size_ratio_threshold, left_table_size / base_size)

        size_range = [1, left_size_ratio_threshold * base_size]

        random_key = np.random.randint(
            key_range[-1] - key_range[0]) + key_range[0]

        random_size = np.random.randint(
            size_range[-1] - size_range[0]) + size_range[0]

        # query_template = f"""
        # DROP VIEW IF EXISTS prev_result_view;
        # CREATE VIEW prev_result_view AS select * from {left_table} ORDER BY RANDOM() LIMIT {random_size};
        # {}
        # EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        # select * from prev_result_view, {base_table} where {base_table_key} = prev_result_view.{left_table_key} and {base_indexed_key} > {random_key};"""

        # self.db.execute(
        #     f"DROP VIEW IF EXISTS prev_result_view; CREATE VIEW prev_result_view AS select * from {left_table} ORDER BY RANDOM() LIMIT {random_size};", set_env=True)

        prev_cte = f'WITH prev_result_view AS (select * from {left_table} ORDER BY RANDOM() LIMIT {random_size}) \n'

        # print("Mixed selectivity", base_indexed_key, base_table_key)
        # if base_indexed_key == base_table_key:
        #     print("Mixed selectivity")
        #     selectivity_query_template = f"""
        #     select count(*) from {base_table} where {base_indexed_key} > {random_key} and {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        # else:
        selectivity_query_template = f"""
        select count(*) from {base_table} where {base_indexed_key} > {random_key};"""
        # print("cardinality sql \n", selectivity_query_template)

        card = self.db.execute(selectivity_query_template)[0][0]

        sel = card / base_size

        # print("sel: ", sel, "left size: ", random_size)

        # rand_query = query_template
        # print(rand_query)

        nl_idx_scan_cost = 0
        nl_seq_scan_cost = 0

        hash_idx_scan_cost = 0
        hash_seq_scan_cost = 0

        merge_idx_scan_cost = 0
        merge_seq_scan_cost = 0

        query_template = f" {prev_cte} select * from prev_result_view, {base_table} where {base_table}.{base_table_key} = prev_result_view.{left_table_key} and {base_table}.{base_indexed_key} > {random_key};"

        # inner_query = f"(select * from {left_table} ORDER BY RANDOM() LIMIT {random_size})"

        # ==============================================
        # merge join + index scan
        # rules = f"/*+\nLeading((prev_result_view {base_table}))\nMergeJoin(prev_result_view {base_table})\nIndexScan({base_table})\n*/\n"
        rules = f"/*+\nMergeJoin(prev_result_view {base_table})\nIndexScan({base_table})\n*/"
        query = rules + \
            'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template
        # print(query)
        # exit(0)
        q = self.db.explain(query)

        # print(q.cardinalities['estimated'])

        merge_idx_scan_cost = q.total_cost
        # ==============================================

        # ==============================================
        # merge join + seq scan
        # rules = f"/*+\nLeading((prev_result_view {base_table}))\nMergeJoin(prev_result_view {base_table})\nSeqScan({base_table})\n*/\n"
        rules = f"/*+\nMergeJoin(prev_result_view {base_table})\nSeqScan({base_table})\n*/"
        query = rules + \
            'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template

        # print(query)
        q = self.db.explain(query)
        merge_seq_scan_cost = q.total_cost
        assert merge_seq_scan_cost != merge_idx_scan_cost
        # exit(0)
        # ==============================================

        # ==============================================
        # hash join + seq scan
        # rules = f"/*+\nLeading((prev_result_view {base_table}))\nHashJoin(prev_result_view {base_table})\nSeqScan({base_table})\n*/\n"
        rules = f"/*+\nHashJoin(prev_result_view {base_table})\nSeqScan({base_table})\n*/\n"
        query = rules + \
            'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template
        q = self.db.explain(query)
        hash_seq_scan_cost = q.total_cost
        # ==============================================

        # ==============================================
        # hash join + index scan
        # rules = f"/*+\nLeading((prev_result_view {base_table}))\nHashJoin(prev_result_view {base_table})\nIndexScan({base_table})\n*/\n"
        rules = f"/*+\nHashJoin(prev_result_view {base_table})\nIndexScan({base_table})\n*/\n"
        query = rules + \
            'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template
        q = self.db.explain(query)
        hash_idx_scan_cost = q.total_cost
        # ==============================================

        # ==============================================
        # nest loop join + idx scan
        # rules = f"/*+\nLeading((prev_result_view {base_table}))\nNestLoop(prev_result_view {base_table})\nIndexScan({base_table})\n*/\n"
        rules = f"/*+\nNestLoop(prev_result_view {base_table})\nIndexScan({base_table})\n*/\n"
        query = rules + \
            'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template
        q = self.db.explain(query)
        nl_idx_scan_cost = q.total_cost
        # ==============================================

        # ==============================================
        # nest loop join + seq scan
        # rules = f"/*+\nLeading((prev_result_view {base_table}))\nNestLoop(prev_result_view {base_table})\nSeqScan({base_table})\n*/\n"
        rules = f"/*+\nNestLoop(prev_result_view {base_table})\nSeqScan({base_table})\n*/\n"
        query = rules + \
            'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template
        q = self.db.explain(query)
        nl_seq_scan_cost = q.total_cost
        # ==============================================

        random_size /= base_size

        # print('EXPLAIN (COSTS) ' + query_template)
        # print("Location: ", sel, random_size)
        # print("costs: ", [nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost, hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost])
        # exit(0)

        return sel, random_size, nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost, hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost

    def parse_costs(self, plan, primary_table):
        pass
        # return scan cost and join cost repectively


class IMDB_lite_query_sampler(QuerySampler):
    def __init__(self):
        super(IMDB_lite_query_sampler, self).__init__()
        self.tables = ['movie_info_idx', 'movie_info',
                       'movie_companies', 'movie_keyword', 'cast_info', 'title']

        self.primary_keys = {
            'movie_info_idx': 'id',
            'movie_info': 'id',
            'movie_companies': 'id',
            'movie_keyword': 'id',
            'cast_info': 'id',
            'title': 'id'
        }  # table: primary key
        self.join_keys = {
            ('title', 'movie_info_idx'): ('id', 'movie_id'),
            ('movie_info_idx', 'title'): ('movie_id', 'id'),

            ('title', 'movie_info'): ('id', 'movie_id'),
            ('movie_info', 'title'): ('movie_id', 'id'),

            ('title', 'movie_companies'): ('id', 'movie_id'),
            ('movie_companies', 'title'): ('movie_id', 'id'),


            ('title', 'movie_keyword'): ('id', 'movie_id'),
            ('movie_keyword', 'title'): ('movie_id', 'id'),

            ('title', 'cast_info'): ('id', 'movie_id'),
            ('cast_info', 'title'): ('movie_id', 'id'),
        }  # (table, table) : (key1, key2)
        self.table_features = {
            'movie_info_idx': {
                'table_size': 1380035,
                'key_range': [1, 1380035]
            },
            'movie_info': {
                'table_size': 14835714,
                'key_range': [1, 14835720]
            },
            'movie_companies': {
                'table_size': 2609129,
                'key_range': [1, 2609129]
            },
            'movie_keyword': {
                'table_size': 4523930,
                'key_range': [1, 4523930]
            },
            'cast_info': {
                'table_size': 36244344,
                'key_range': [1, 36244344]
            },
            'title': {
                'table_size': 2528312,
                'key_range': [1, 2528312]
            }
        }  # table features

        self.db = Database(db_name='imdb')  # the database used by sampler

        self.left_size_ratio_threshold = 0.5

        self.join_graph = {
            # constructed by base table + left table list
            'movie_info_idx': ['title'],
            'movie_info': ['title'],
            'movie_companies': ['title'],
            'movie_keyword': ['title'],
            'cast_info': ['title'],
            'title': ['movie_info_idx', 'movie_info',
                      'movie_companies', 'movie_keyword', 'cast_info']
        }


class TPCH_query_sampler(QuerySampler):

    def __init__(self):
        super(TPCH_query_sampler, self).__init__()
        self.tables = ['part', 'lineitem', 'supplier', 'orders']

        self.primary_keys = {
            'part': 'p_partkey',
            'supplier': 's_suppkey',
            'orders': 'o_orderkey',
            'customer': 'c_custkey',
            'partsupp': 'ps_partkey'
        }  # table: primary key
        self.join_keys = {
            ('part', 'lineitem'): ('p_partkey', 'l_partkey'),
            ('lineitem', 'part'): ('l_partkey', 'p_partkey'),

            ('orders', 'lineitem'): ('o_orderkey', 'l_orderkey'),
            ('lineitem', 'orders'): ('l_orderkey', 'o_orderkey'),

            ('orders', 'customer'): ('o_custkey', 'c_custkey'),
            ('customer', 'orders'): ('c_custkey', 'o_custkey'),

            ('supplier', 'lineitem'): ('s_suppkey', 'l_suppkey'),
            ('lineitem', 'supplier'): ('l_suppkey', 's_suppkey'),

            ('supplier', 'partsupp'): ('s_suppkey', 'ps_suppkey'),
            ('partsupp', 'supplier'): ('ps_suppkey', 's_suppkey'),

            ('part', 'partsupp'): ('p_partkey', 'ps_partkey'),
            ('partsupp', 'part'): ('ps_partkey', 'p_partkey'),

            ('lineitem', 'partsupp'): ('l_partkey', 'ps_partkey'),
            ('partsupp', 'lineitem'): ('ps_partkey', 'l_partkey'),


        }  # (table, table) : (key1, key2)
        self.table_features = {
            'part': {
                'table_size': 200000,
                'key_range': [1, 200000]
            },
            'lineitem': {
                'table_size': 6000003,
                'key_range': []
            },
            'orders': {
                'table_size': 1500000,
                'key_range': [1, 6000000]
            },
            'supplier': {
                'table_size': 10000,
                'key_range': [1, 10000]
            },
            'customer': {
                'table_size': 150000,
                'key_range': [1, 150000]
            },
            'partsupp': {
                'table_size': 800000,
                'key_range': [1, 200000]
            }
        }  # table features

        self.db = Database(db_name='tpch')  # the database used by sampler

        self.left_size_ratio_threshold = 0.5

        self.join_graph = {
            # constructed by base table + left table list
            # 'part': ['lineitem', 'partsupp'],
            'orders': ['lineitem', 'customer'],
            # 'supplier': ['lineitem', 'partsupp'],
            # 'customer': ['orders'],
            # 'partsupp': ['lineitem', 'supplier'],
        }


class SSB_query_sampler(QuerySampler):

    def __init__(self):
        super(SSB_query_sampler, self).__init__()
        self.tables = ['part', 'lineitem', 'supplier', 'orders']

        self.primary_keys = {
            'part': 'p_partkey',
            'supplier': 's_suppkey',
            'ddate': 'd_datekey',
            'customer': 'c_custkey'
        }  # table: primary key
        self.join_keys = {
            ('lineorder', 'customer'): ('lo_custkey', 'c_custkey'),
            ('customer', 'lineorder'): ('c_custkey', 'lo_custkey'),

            ('supplier', 'lineorder'): ('s_suppkey', 'lo_suppkey'),
            ('lineorder', 'supplier'): ('lo_suppkey', 's_suppkey'),

            ('part', 'lineorder'): ('p_part', 'lo_partkey'),
            ('lineorder', 'part'): ('lo_partkey', 'p_partkey'),

            ('ddate', 'lineorder'): ('d_datekey', 'lo_orderdate'),
            ('lineorder', 'ddate'): ('lo_orderdate', 'd_datekey'),

        }  # (table, table) : (key1, key2)

        self.table_features = {
            'part': {
                'table_size': 200000,
                'key_range': [1, 200000]
            },
            'supplier': {
                'table_size': 2000,
                'key_range': [1, 2000]
            },
            'ddate': {
                'table_size': 2556,
                'key_range': [19920101,  19981230]
            },
            'customer': {
                'table_size': 30000,
                'key_range': [1,  30000]
            },
            'lineorder': {
                'table_size':  6001171,
                'key_range': []
            }
        }  # table features

        self.db = Database(db_name='ssb')  # the database used by sampler

        self.left_size_ratio_threshold = 0.5

        self.join_graph = {
            # constructed by base table + left table list
            'part': ['lineorder'],
            'customer': ['lineorder'],
            'ddate': ['lineorder'],
            'supplier': ['lineorder']
        }


def visualize_pair_on_dataset(db_name='tpch'):
    if db_name.lower() == 'tpch':
        sampler = TPCH_query_sampler()
    elif db_name.lower() == 'imdb':
        sampler = IMDB_lite_query_sampler()
    elif db_name.lower() == 'ssb':
        sampler = SSB_query_sampler()

    # sampler.left_size_ratio_threshold = 0.01

    for base_table in sampler.join_graph:
        for left_table in sampler.join_graph[base_table]:

            res = sampler.sample_for_table(
                base_table, [left_table], sample_size=500)

            viz = DecisionVisualizer()
            viz.plot_2d_optimal_decision_with_importance(res, title=f"Optimal operator (left: {left_table}, base: {base_table})", filename=f"{left_table}_{base_table}_optimal",
                                                         base_dir=f'./figures/{db_name.lower()}/{base_table}/')
            exit(1)


if __name__ == "__main__":
    # visualize_pair_on_dataset(db_name='ssb')
    visualize_pair_on_dataset(db_name='tpch')
    # visualize_pair_on_dataset(db_name='imdb')
