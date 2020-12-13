import numpy as np
import pandas as pd
# from cardinality_estimation_quality.cardinality_estimation_quality import *
import os
import sys
from Visualizer import *
from tqdm import tqdm
from DB_connector import *
from Schema import *
import itertools


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

    def __init__(self, left_size_ratio_threshold=0.5):
        self.join_graph = None
        self.tables = None
        self.primary_keys = {}  # table: primary key
        self.join_keys = {}  # (table, table) : (key1, key2)
        self.table_features = {}  # table features
        self.db = None  # the database used by sampler
        self.left_size_ratio_threshold = left_size_ratio_threshold
        self.max_predicate_num = 3

    def sample_for_table(self, primary_table, left_tables=[], sample_size=100, sample_with_replacement=False, left_order=None):
        results = []
        # ==============================================
        # Manual seed to make it reproducible
        print(f"Set numpy random seed 0 for reproducibility")
        np.random.seed(0)
        # ==============================================
        print(
            f"{self.db.db_name}: Samping for {primary_table}, with left tables: {left_tables}, left table sorted on {left_order}")
        for _ in tqdm(range(sample_size)):
            rand_table = left_tables[np.random.randint(len(left_tables))]
            results.append(self.collect_random_query_2_table_all_op(
                primary_table, rand_table, sample_with_replacement, left_order=left_order))
        return results

    def collect_random_query_2_table_all_op(self, base_table, left_table, sample_with_replacement, left_order='random'):
        # Overwritten by child classes
        pass


class Postgres_QuerySampler(QuerySampler):

    def __init__(self, db_name='imdb', left_size_ratio_threshold=0.5, schema=None):
        super(Postgres_QuerySampler, self).__init__(
            left_size_ratio_threshold=left_size_ratio_threshold)
        self.schema = schema
        self.db = Postgres_Connector(db_name=db_name)

    def collect_random_query_2_table_all_op(self, base_table, left_table, sample_with_replacement, left_order=None):

        # ==============================================
        # load features
        base_size = self.schema.table_features[base_table]['table_size']
        key_range = self.schema.table_features[base_table]['key_range']
        left_table_size = self.schema.table_features[left_table]['table_size']
        left_table_key, base_table_key = self.schema.join_keys[(
            left_table, base_table)]
        non_indexed_attr, non_indexed_attr_range = self.schema.non_indexed_attr[base_table]
        base_indexed_key = self.schema.primary_keys[base_table]
        left_indexed_key = self.schema.primary_keys[left_table] if left_table in self.schema.primary_keys.keys(
        ) else None
        if left_order is None:
            left_order = ""
        elif left_order.lower() == 'random':
            left_order = "ORDER BY RANDOM()"
        elif left_order.lower() == 'left_join_key':
            left_order = f"ORDER BY {left_table_key}"
        else:
            exit(f"Left order {left_order} is not supported!")
        # ==============================================

        if self.left_size_ratio_threshold < 0:
            left_size_ratio_threshold = left_table_size / base_size
        else:
            left_size_ratio_threshold = min(
                self.left_size_ratio_threshold, left_table_size / base_size)

        size_range = [1, left_size_ratio_threshold * base_size]

        random_size = np.random.randint(
            size_range[-1] - size_range[0]) + size_range[0]

        # ==============================================
        # Generate random predicate on indexed attribute
        random_indexed_predicate_num = np.random.randint(
            self.max_predicate_num) + 1
        predicates_on_indexed_attr_list = []
        for _ in range(random_indexed_predicate_num):
            random_key = self.schema.random_indexed_attr_value(base_table)
            predicates_on_indexed_attr_list.append(
                f'{base_table}.{base_indexed_key} { np.random.choice([" > "]) } {random_key}')
        predicates_on_indexed_attr = ' and '.join(
            predicates_on_indexed_attr_list)
        # ==============================================

        # ==============================================
        # Generate random predicate on non-indexed attribute
        random_non_indexed_predicate_num = np.random.randint(
            self.max_predicate_num)
        predicates_on_non_indexed_attr_list = []

        for _ in range(random_non_indexed_predicate_num):
            random_attr = self.schema.random_non_indexed_attr_value(base_table)
            predicates_on_non_indexed_attr_list.append(
                f'{base_table}.{non_indexed_attr} { np.random.choice([" > "]) } {random_attr}')
        all_predicates_on_base_table_list = predicates_on_non_indexed_attr_list + \
            predicates_on_indexed_attr_list
        predicates_on_non_indexed_attr = ' and '.join(
            predicates_on_non_indexed_attr_list)
        # ==============================================

        if not sample_with_replacement:
            prev_cte = f'WITH prev_result_view AS (select * from {left_table} {left_order} LIMIT {random_size}) \n'
        else:
            prev_cte = f"""
            WITH prev_result_view AS (with rows as (select *,row_number() over() as rn from {left_table} order by random()),
            w(num) as (select (random()*(select count(*) from rows))::int+1
                from generate_series(1,{random_size}))
            select rows.* from rows join w on rows.rn = w.num {left_order} LIMIT {random_size})\n"""

        # ==============================================
        # Selectivity of predicate on indexed attr
        selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        select * from {base_table} where {predicates_on_indexed_attr};"""
        q = self.db.explain(selectivity_query_template)
        sel_of_pred_on_indexed_attr = q.cardinalities['estimated'][0] / base_size
        # ==============================================

        # ==============================================
        # Selectivity on non_indexed attr
        if random_non_indexed_predicate_num > 0:
            selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
            select * from {base_table} where {predicates_on_non_indexed_attr};"""
            q = self.db.explain(selectivity_query_template)
            sel_of_pred_on_non_indexed_attr = q.cardinalities['estimated'][0] / base_size
        else:
            sel_of_pred_on_non_indexed_attr = 1
        # ==============================================

        # ==============================================
        # Selectivity of pred on non-indexed and indexed attrs on base table
        selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        select * from {base_table} where {' and '.join(all_predicates_on_base_table_list)};"""
        q = self.db.explain(selectivity_query_template)
        sel_of_pred_on_indexed_attr_and_non_indexed_attr = q.cardinalities[
            'estimated'][0] / base_size
        # ==============================================

        # ==============================================
        # Join predicate selectivity
        selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        {prev_cte} select * from {base_table} where {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        q = self.db.explain(
            selectivity_query_template)
        sel_of_join_pred = q.cardinalities['estimated'][0] / base_size
        # ==============================================

        # ==============================================
        # Selectivity on indexed attr include join predicate
        selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        {prev_cte} select * from {base_table} where {predicates_on_indexed_attr} and {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        q = self.db.explain(
            selectivity_query_template)
        sel_of_pred_on_indexed_attr_and_join_pred = q.cardinalities['estimated'][0] / base_size
        # ==============================================

        # ==============================================
        # Selectivity on non indexed attr include join predicate
        selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        {prev_cte} select * from {base_table} where {' and '.join(predicates_on_non_indexed_attr_list + [f'{base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view)'])};"""
        q = self.db.explain(
            selectivity_query_template)
        sel_of_pred_on_non_indexed_attr_and_join_pred = q.cardinalities[
            'estimated'][0] / base_size
        # ==============================================

        # ==============================================
        # Total sel on base table
        selectivity_query_template = f"""EXPLAIN (COSTS, VERBOSE, FORMAT JSON)
        {prev_cte} select * from {base_table} where {' and '.join(all_predicates_on_base_table_list)} and {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        q = self.db.explain(
            selectivity_query_template)
        total_sel_on_base_table = q.cardinalities['estimated'][0] / base_size
        # ==============================================

        # ==============================================
        # Run query with different physical operators
        query_template = f" {prev_cte} select * from prev_result_view, {base_table} where {' and '.join([f'{base_table}.{base_table_key} = prev_result_view.{left_table_key}'] + all_predicates_on_base_table_list)};"
        costs = {}
        cte_costs = {}
        queries = {}
        for join_method, scan_method in itertools.product(['MergeJoin', 'HashJoin', 'NestLoop'], ['SeqScan', 'IndexScan']):
            rules = f"/*+\n{join_method}(prev_result_view {base_table})\n{scan_method}({base_table})\n*/"
            query = rules + \
                'EXPLAIN (COSTS, VERBOSE, FORMAT JSON) ' + query_template
            q = self.db.explain(query)
            queries[(join_method, scan_method)] = query
            costs[(join_method, scan_method)
                  ], cte_costs[(join_method, scan_method)] = postgres_triplet_cost_parse(q)

        nl_idx_scan_cost, nl_idx_scan_cte_cost = costs[(
            'NestLoop', 'IndexScan')], cte_costs[('NestLoop', 'IndexScan')]
        nl_seq_scan_cost, nl_seq_scan_cte_cost = costs[(
            'NestLoop', 'SeqScan')], cte_costs[('NestLoop', 'SeqScan')]

        hash_idx_scan_cost, hash_idx_scan_cte_cost = costs[(
            'HashJoin', 'IndexScan')], cte_costs[('HashJoin', 'IndexScan')]
        hash_seq_scan_cost, hash_seq_scan_cte_cost = costs[(
            'HashJoin', 'SeqScan')], cte_costs[('HashJoin', 'SeqScan')]

        merge_idx_scan_cost, merge_idx_scan_cte_cost = costs[(
            'MergeJoin', 'IndexScan')], cte_costs[('MergeJoin', 'IndexScan')]
        merge_seq_scan_cost, merge_seq_scan_cte_cost = costs[(
            'MergeJoin', 'SeqScan')], cte_costs[('MergeJoin', 'SeqScan')]

        nl_idx_scan_query = queries[('NestLoop', 'IndexScan')]
        nl_seq_scan_query = queries[('NestLoop', 'SeqScan')]

        hash_idx_scan_query = queries[('HashJoin', 'IndexScan')]
        hash_seq_scan_query = queries[('HashJoin', 'SeqScan')]

        merge_idx_scan_query = queries[('MergeJoin', 'IndexScan')]
        merge_seq_scan_query = queries[('MergeJoin', 'SeqScan')]
        # ==============================================

        # ==============================================
        # collect all the features needed for training
        # features included:
        # left cardinality; base cardinality; left ordered on join key; base ordered on joined key; selectivity on indexed attr;
        left_ratio = random_size / base_size
        features = {}
        features['query'] = query_template

        features['hj_idx_query'] = hash_idx_scan_query
        features['hj_seq_query'] = hash_seq_scan_query
        features['nl_idx_query'] = nl_idx_scan_query
        features['nl_seq_query'] = nl_seq_scan_query
        features['mj_idx_query'] = merge_idx_scan_query
        features['mj_seq_query'] = merge_seq_scan_query

        features['left_cardinality'] = random_size
        features['left_cardinality_ratio'] = left_ratio
        features['base_cardinality'] = base_size
        features['left_ordered'] = 1 if left_indexed_key == left_table_key else 0
        features['base_ordered'] = 1 if base_table_key == base_indexed_key else 0
        features['index_size'] = self.schema.indexes[base_table][base_indexed_key]
        features['result_size'] = q.cardinalities['estimated'][0]

        features['sel_of_pred_on_indexed_attr'] = sel_of_pred_on_indexed_attr
        features['sel_of_pred_on_indexed_attr_and_join_pred'] = sel_of_pred_on_indexed_attr_and_join_pred
        features['sel_of_pred_on_non_indexed_attr_and_join_pred'] = sel_of_pred_on_non_indexed_attr_and_join_pred
        features['sel_of_join_pred'] = sel_of_join_pred
        features['sel_of_pred_on_non_indexed_attr'] = sel_of_pred_on_non_indexed_attr
        features['sel_of_pred_on_indexed_attr_and_non_indexed_attr'] = sel_of_pred_on_indexed_attr_and_non_indexed_attr
        features['total_sel_on_base_table'] = total_sel_on_base_table

        features['predicate_op_num_on_indexed_attr'] = random_indexed_predicate_num
        features['predicate_op_num_on_non_indexed_attr'] = random_non_indexed_predicate_num

        features['hj_idx_cost'] = hash_idx_scan_cost
        features['hj_seq_cost'] = hash_seq_scan_cost
        features['nl_idx_cost'] = nl_idx_scan_cost
        features['nl_seq_cost'] = nl_seq_scan_cost
        features['mj_idx_cost'] = merge_idx_scan_cost
        features['mj_seq_cost'] = merge_seq_scan_cost

        features['hj_idx_cte_cost'] = hash_idx_scan_cte_cost
        features['hj_seq_cte_cost'] = hash_seq_scan_cte_cost
        features['nl_idx_cte_cost'] = nl_idx_scan_cte_cost
        features['nl_seq_cte_cost'] = nl_seq_scan_cte_cost
        features['mj_idx_cte_cost'] = merge_idx_scan_cte_cost
        features['mj_seq_cte_cost'] = merge_seq_scan_cte_cost

        features['optimal_decision'] = np.argmin(
            [hash_idx_scan_cost, hash_seq_scan_cost, nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost])

        features['visualization_features'] = (sel_of_pred_on_indexed_attr, left_ratio, hash_idx_scan_cost, hash_seq_scan_cost,
                                      nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost)
        # ==============================================
        # print(features)
        return features


class Mssql_QuerySampler(QuerySampler):

    def __init__(self, db_name='imdb', left_size_ratio_threshold=0.5, schema=None):
        super(Mssql_QuerySampler, self).__init__(
            left_size_ratio_threshold=left_size_ratio_threshold)
        self.schema = schema
        self.db = Mssql_Connector(db_name=db_name)

    def collect_random_query_2_table_all_op(self, base_table, left_table, sample_with_replacement=False, left_order=None):
        # ==============================================
        # load features
        base_size = self.schema.table_features[base_table]['table_size']
        key_range = self.schema.table_features[base_table]['key_range']
        left_table_size = self.schema.table_features[left_table]['table_size']
        non_indexed_attr, non_indexed_attr_range = self.schema.non_indexed_attr[base_table]
        left_table_key, base_table_key = self.schema.join_keys[(
            left_table, base_table)]
        base_indexed_key = self.schema.primary_keys[base_table]
        left_indexed_key = self.schema.primary_keys[left_table] if left_table in self.schema.primary_keys.keys(
        ) else None

        if left_order is None:
            left_order = ""
        elif left_order.lower() == 'random':
            left_order = "ORDER BY NEWID()"
        elif left_order.lower() == 'left_join_key':
            left_order = f"ORDER BY {left_table_key}"
        else:
            exit(f"Left order {left_order} is not supported!")

        # ==============================================

        # ==============================================
        # Generate random left size
        if self.left_size_ratio_threshold < 0:
            left_size_ratio_threshold = left_table_size / base_size
        else:
            left_size_ratio_threshold = min(
                self.left_size_ratio_threshold, left_table_size / base_size)

        size_range = [1, left_size_ratio_threshold * base_size]
        random_size = np.random.randint(
            size_range[-1] - size_range[0]) + size_range[0]
        # ==============================================

        # ==============================================
        # Generate random predicate on indexed attribute
        random_indexed_predicate_num = np.random.randint(
            self.max_predicate_num) + 1
        predicates_on_indexed_attr_list = []
        for _ in range(random_indexed_predicate_num):
            random_key = self.schema.random_indexed_attr_value(base_table)
            predicates_on_indexed_attr_list.append(
                f'{base_table}.{base_indexed_key} { np.random.choice([" > "]) } {random_key}')
        predicates_on_indexed_attr = ' and '.join(
            predicates_on_indexed_attr_list)
        # ==============================================

        # ==============================================
        # Generate random predicate on non-indexed attribute
        random_non_indexed_predicate_num = np.random.randint(
            self.max_predicate_num)
        predicates_on_non_indexed_attr_list = []

        for _ in range(random_non_indexed_predicate_num):
            random_attr = self.schema.random_non_indexed_attr_value(base_table)
            predicates_on_non_indexed_attr_list.append(
                f'{base_table}.{non_indexed_attr} { np.random.choice([" > "]) } {random_attr}')
        all_predicates_on_base_table_list = predicates_on_non_indexed_attr_list + \
            predicates_on_indexed_attr_list
        predicates_on_non_indexed_attr = ' and '.join(
            predicates_on_non_indexed_attr_list)
        # ==============================================

        if not sample_with_replacement:
            prev_cte = f'WITH prev_result_view AS (select TOP {random_size} * from {left_table} {left_order}) \n'
        else:
            print("Sample with replacement for sql server is not supported yet")
            exit(1)

        # ==============================================
        # Selectivity on indexed attribute
        selectivity_query_template = f"""select * from {base_table} where {predicates_on_indexed_attr};"""
        q = self.db.explain(selectivity_query_template)
        sel_of_pred_on_indexed_attr = q['estimated_result_size'] / base_size
        # ==============================================

        # ==============================================
        # Selectivity on non indexed attribute
        if random_non_indexed_predicate_num > 0:
            selectivity_query_template = f"""select * from {base_table} where {predicates_on_non_indexed_attr};"""
            q = self.db.explain(selectivity_query_template)
            sel_of_pred_on_non_indexed_attr = q['estimated_result_size'] / base_size
        else:
            sel_of_pred_on_non_indexed_attr = 1
        # ==============================================

        # ==============================================
        # Selectivity of join predicate on base table
        selectivity_query_template = f"""{prev_cte}
        select * from {base_table} where {predicates_on_indexed_attr} and {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        q = self.db.explain(
            selectivity_query_template)
        sel_of_join_pred = q['estimated_result_size'] / base_size
        # ==============================================

        # ==============================================
        # selectivity of predicate on indexed attr and non indexed attr on base table
        selectivity_query_template = f"""select * from {base_table} where {' and '.join(all_predicates_on_base_table_list)};"""
        q = self.db.explain(selectivity_query_template)
        sel_of_pred_on_indexed_attr_and_non_indexed_attr = q['estimated_result_size'] / base_size
        # ==============================================

        # ==============================================
        # Selectivity of join predicate on base table
        selectivity_query_template = f"""{prev_cte}
        select * from {base_table} where {' and '.join(all_predicates_on_base_table_list)} and {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        q = self.db.explain(
            selectivity_query_template)
        total_sel_on_base_table = q['estimated_result_size'] / base_size
        # ==============================================

        # ==============================================
        # Selectivity of (predicate_on_indexed_attr and join_pred)
        selectivity_query_template = f"""{prev_cte}
        select * from {base_table} where {predicates_on_indexed_attr} and {base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view);"""
        q = self.db.explain(
            selectivity_query_template)
        sel_of_pred_on_indexed_attr_and_join_pred = q['estimated_result_size'] / base_size
        # ==============================================

        # ==============================================
        # Selectivity of (predicate_on_non_indexed_attr and join_pred)
        selectivity_query_template = f"""{prev_cte}
        select * from {base_table} where {' and '.join(predicates_on_non_indexed_attr_list + [f'{base_table}.{base_indexed_key} in (select {left_table_key} from prev_result_view)'])};"""
        q = self.db.explain(
            selectivity_query_template)
        sel_of_pred_on_non_indexed_attr_and_join_pred = q['estimated_result_size'] / base_size
        # ==============================================

        index_name = base_indexed_key.split('_')[-1]

        query_template = f"""{prev_cte} select * from prev_result_view, {base_table} %s where {' and '.join([f'{base_table}.{base_table_key} = prev_result_view.{left_table_key}'] + all_predicates_on_base_table_list)} %s;"""

        costs = {}
        cte_costs = {}
        queries = {}

        for join_method, scan_method in itertools.product(['MERGE', 'LOOP', 'HASH'], [f'INDEX({"PK_" + base_table}) FORCESEEK', 'INDEX(0) FORCESCAN']):
            rules = (f"WITH ({scan_method}) ",
                     f" OPTION ({join_method} JOIN, MAXDOP 1, NO_PERFORMANCE_SPOOL, use hint('DISABLE_BATCH_MODE_ADAPTIVE_JOINS') )")
            query = query_template % rules
            q = self.db.explain(query)
            # print(query)
            queries[(join_method, scan_method)] = query
            costs[(join_method, scan_method)] = q['total_estimated_cost']

            if q['left_table'] == base_table:
                cte_costs[(join_method, scan_method)] = q['right_cost']
                # useless_costs.append(q['right_cost'])
            else:
                cte_costs[(join_method, scan_method)] = q['left_cost']
                # useless_costs.append(q['left_cost'])

        nl_idx_scan_cost = costs[(
            'LOOP', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        nl_idx_scan_cte_cost = cte_costs[(
            'LOOP', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        nl_seq_scan_cost = costs[('LOOP', f'INDEX(0) FORCESCAN')]
        nl_seq_scan_cte_cost = cte_costs[('LOOP', f'INDEX(0) FORCESCAN')]

        hash_idx_scan_cost = costs[(
            'HASH', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        hash_idx_scan_cte_cost = cte_costs[(
            'HASH', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        hash_seq_scan_cost = costs[('HASH', f'INDEX(0) FORCESCAN')]
        hash_seq_scan_cte_cost = cte_costs[('HASH', f'INDEX(0) FORCESCAN')]

        merge_idx_scan_cost = costs[(
            'MERGE', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        merge_idx_scan_cte_cost = cte_costs[(
            'MERGE', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        merge_seq_scan_cost = costs[('MERGE', f'INDEX(0) FORCESCAN')]
        merge_seq_scan_cte_cost = cte_costs[('MERGE', f'INDEX(0) FORCESCAN')]

        nl_idx_scan_query = queries[(
            'LOOP', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        nl_seq_scan_query = costs[('LOOP', f'INDEX(0) FORCESCAN')]

        hash_idx_scan_query = queries[(
            'HASH', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        hash_seq_scan_query = queries[('HASH', f'INDEX(0) FORCESCAN')]

        merge_idx_scan_query = queries[(
            'MERGE', f'INDEX({"PK_" + base_table}) FORCESEEK')]
        merge_seq_scan_query = queries[('MERGE', f'INDEX(0) FORCESCAN')]

        # ==============================================
        # collect all the features needed for training
        # features included:
        # left cardinality; base cardinality; left ordered on join key; base ordered on joined key; selectivity on indexed attr;

        features = {}
        left_ratio = random_size / base_size
        features['query'] = query_template % ('', '')

        features['hj_idx_query'] = hash_idx_scan_query
        features['hj_seq_query'] = hash_seq_scan_query
        features['nl_idx_query'] = nl_idx_scan_query
        features['nl_seq_query'] = nl_seq_scan_query
        features['mj_idx_query'] = merge_idx_scan_query
        features['mj_seq_query'] = merge_seq_scan_query

        features['left_cardinality'] = random_size
        features['left_cardinality_ratio'] = left_ratio
        features['base_cardinality'] = base_size

        features['left_ordered'] = 1 if left_indexed_key == left_table_key else 0
        features['base_ordered'] = 1 if base_table_key == base_indexed_key else 0
        features['index_size'] = self.schema.indexes[base_table][base_indexed_key]
        features['result_size'] = q['estimated_result_size']

        features['sel_of_pred_on_indexed_attr'] = sel_of_pred_on_indexed_attr
        features['sel_of_pred_on_indexed_attr_and_join_pred'] = sel_of_pred_on_indexed_attr_and_join_pred
        features['sel_of_pred_on_non_indexed_attr_and_join_pred'] = sel_of_pred_on_non_indexed_attr_and_join_pred
        features['sel_of_join_pred'] = sel_of_join_pred
        features['sel_of_pred_on_non_indexed_attr'] = sel_of_pred_on_non_indexed_attr
        features['sel_of_pred_on_indexed_attr_and_non_indexed_attr'] = sel_of_pred_on_indexed_attr_and_non_indexed_attr
        features['total_sel_on_base_table'] = total_sel_on_base_table

        features['predicate_op_num_on_indexed_attr'] = random_indexed_predicate_num
        features['predicate_op_num_on_non_indexed_attr'] = random_non_indexed_predicate_num

        features['hj_idx_cost'] = hash_idx_scan_cost
        features['hj_seq_cost'] = hash_seq_scan_cost
        features['nl_idx_cost'] = nl_idx_scan_cost
        features['nl_seq_cost'] = nl_seq_scan_cost
        features['mj_idx_cost'] = merge_idx_scan_cost
        features['mj_seq_cost'] = merge_seq_scan_cost

        features['hj_idx_cte_cost'] = hash_idx_scan_cte_cost
        features['hj_seq_cte_cost'] = hash_seq_scan_cte_cost
        features['nl_idx_cte_cost'] = nl_idx_scan_cte_cost
        features['nl_seq_cte_cost'] = nl_seq_scan_cte_cost
        features['mj_idx_cte_cost'] = merge_idx_scan_cte_cost
        features['mj_seq_cte_cost'] = merge_seq_scan_cte_cost

        features['optimal_decision'] = np.argmin(
            [hash_idx_scan_cost, hash_seq_scan_cost, nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost])
        features['visualization_features'] = (sel_of_pred_on_indexed_attr, left_ratio, hash_idx_scan_cost, hash_seq_scan_cost,
                                                nl_idx_scan_cost, nl_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost)
        # features['visualization_features'] = (q.cardinalities['estimated'][0], left_ratio, nl_idx_scan_cost, nl_seq_scan_cost,
        #                                       hash_idx_scan_cost, hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost)
        # ==============================================
        # print(features)
        return features
        # return sel_on_indexed_attr, left_ratio, , nl_idx_scan_cost, nl_seq_scan_cost, hash_idx_scan_cost, hash_seq_scan_cost, merge_idx_scan_cost, merge_seq_scan_cost


class IMDB_lite_QuerySampler:

    def __init__(self, db_engine='postgres', left_size_ratio_threshold=0.5):

        self.schema = IMDB_lite_schema()

        if db_engine == 'postgres':
            db_connector = Postgres_Connector
            self.query_sampler = Postgres_QuerySampler(db_name='imdb',
                                                       left_size_ratio_threshold=left_size_ratio_threshold, schema=self.schema)
        elif db_engine == 'mssql':
            # db_connector = Postgres_Connector
            self.query_sampler = Mssql_QuerySampler(db_name='imdb',
                                                    left_size_ratio_threshold=left_size_ratio_threshold, schema=self.schema)
            # self.query_sampler.db = db_connector(db_name='imdb')
        else:
            exit(f"db_engine = {db_engine} is not supported yet")


class TPCH_QuerySampler:

    def __init__(self, db_engine='postgres', left_size_ratio_threshold=0.5):

        self.schema = TPCH_Schema()

        if db_engine == 'postgres':
            self.query_sampler = Postgres_QuerySampler(db_name='tpch',
                                                       left_size_ratio_threshold=left_size_ratio_threshold, schema=self.schema)
        elif db_engine == 'mssql':
            self.query_sampler = Mssql_QuerySampler(db_name='tpch',
                                                    left_size_ratio_threshold=left_size_ratio_threshold, schema=self.schema)
        else:
            exit(f"db = {db_engine} is not supported yet")


class SSB_QuerySampler:

    def __init__(self, db_engine='postgres', left_size_ratio_threshold=0.5):

        self.schema = SSB_Schema()

        if db_engine == 'postgres':
            db_connector = Postgres_Connector
            self.query_sampler = Postgres_QuerySampler(db_name='ssb',
                                                       left_size_ratio_threshold=left_size_ratio_threshold, schema=self.schema)
        elif db_engine == 'mssql':
            self.query_sampler = Mssql_QuerySampler(db_name='ssb',
                                                    left_size_ratio_threshold=left_size_ratio_threshold, schema=self.schema)
        else:
            exit(f"db = {db_engine} is not supported yet")


def visualize_pair_on_dataset(db_name='tpch', sample_with_replacement=False):

    if db_name.lower() == 'tpch':
        sampler = TPCH_QuerySampler(db_engine='postgres')
    elif db_name.lower() == 'imdb':
        sampler = IMDB_lite_QuerySampler(db_engine='postgres')
    elif db_name.lower() == 'ssb':
        sampler = SSB_QuerySampler(db_engine='postgres')

    # sampler.left_size_ratio_threshold = 0.01

    for base_table in sampler.schema.join_graph:
        for left_table in sampler.schema.join_graph[base_table]:
            for left_order in ['random']:

                res = sampler.query_sampler.sample_for_table(
                    base_table, [left_table], sample_size=500, sample_with_replacement=sample_with_replacement, left_order=left_order)

                if sample_with_replacement:
                    fig_filename = f"{left_table}_{base_table}_optimal_rep"
                    data_filename = f"{left_table}_{base_table}_optimal_rep.csv"
                else:
                    fig_filename = f"{left_table}_{base_table}_optimal"
                    data_filename = f"{left_table}_{base_table}_optimal.csv"

                viz = DecisionVisualizer()
                viz.plot_2d_optimal_decision_with_importance(res, title=f"Optimal operator (left: {left_table}, base: {base_table})", filename=fig_filename,
                                                             base_dir=f'./figures/{db_name.lower()}/{base_table}/')
            # exit(0)

                # save_data(res, save_path=f'./data/{db_name.lower()}/{base_table}/',
                #           filename=data_filename, column_names=['left_cardinality', 'left_cardinality_ratio', 'base_cardinality', 'selectivity_on_indexed_attr', 'left_ordered', 'base_ordered',
                #                                                 'hj_idx_cost', 'hj_seq_cost', 'nl_idx_cost', 'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost', 'optimal_decision'])


def prepare_data_on_dataset(db_name='tpch', db_engine='postgres', sample_with_replacement=False):

    if db_name.lower() == 'tpch':
        sampler = TPCH_QuerySampler(db_engine=db_engine)
    elif db_name.lower() == 'imdb':
        sampler = IMDB_lite_QuerySampler(db_engine=db_engine)
    elif db_name.lower() == 'ssb':
        sampler = SSB_QuerySampler(db_engine=db_engine)

    result = []

    for base_table in sampler.schema.join_graph:
        for left_table in sampler.schema.join_graph[base_table]:
            result = []

            for left_order in ['random']:

                res = sampler.query_sampler.sample_for_table(
                    base_table, [left_table], sample_size=2000, sample_with_replacement=sample_with_replacement, left_order=left_order)
                result += res

            if sample_with_replacement:
                data_filename = f"{db_engine}_{left_table}_{base_table}_optimal_rep.csv"
            else:
                data_filename = f"{db_engine}_{left_table}_{base_table}_optimal.csv"

            save_data(result, save_path=f'./sample_results/{db_name.lower()}/{base_table}/',
                      filename=data_filename, column_names=['query', 'hj_idx_query', 'hj_seq_query', 'nl_idx_query', 'nl_seq_query', 'mj_idx_query', 'mj_seq_query',
                                                            'left_cardinality', 'left_cardinality_ratio', 'base_cardinality',
                                                            'sel_of_join_pred', 'sel_of_pred_on_indexed_attr', 'sel_of_pred_on_non_indexed_attr',
                                                            'sel_of_pred_on_indexed_attr_and_join_pred',
                                                            'sel_of_pred_on_non_indexed_attr_and_join_pred', 'sel_of_pred_on_indexed_attr_and_non_indexed_attr',
                                                            'total_sel_on_base_table',
                                                            'left_ordered', 'base_ordered', 'result_size', 'predicate_op_num_on_indexed_attr',
                                                            'predicate_op_num_on_non_indexed_attr',
                                                            'hj_idx_cost', 'hj_seq_cost', 'nl_idx_cost', 'nl_seq_cost', 'mj_idx_cost', 'mj_seq_cost',
                                                            'hj_idx_cte_cost', 'hj_seq_cte_cost', 'nl_idx_cte_cost', 'nl_seq_cte_cost', 'mj_idx_cte_cost', 'mj_seq_cte_cost',
                                                            'optimal_decision'])


def save_data(results, column_names, save_path, filename):

    data = []

    for f in results:
        curr = []
        for c in column_names:
            curr.append(f[c])
        data.append(curr)

    df = pd.DataFrame(data=data, columns=column_names)
    df.to_csv(os.path.join(save_path, filename), index=False)
    # print("Saving data to ", os.path.join(save_path, filename))
    # exit(1)


if __name__ == "__main__":

    import itertools

    # visualize_pair_on_dataset(db_name='ssb', sample_with_replacement=True)
    # visualize_pair_on_dataset(db_name='ssb', sample_with_replacement=False)

    # visualize_pair_on_dataset(db_name='imdb', sample_with_replacement=False)

    # visualize_pair_on_dataset(db_name='tpch', sample_with_replacement=True)
    # visualize_pair_on_dataset(db_name='tpch', sample_with_replacement=False)

    for db_engine, db_name, sample_w_rep in itertools.product(['postgres'], ['imdb'], [False]):
        prepare_data_on_dataset(db_engine=db_engine,
                                db_name=db_name, sample_with_replacement=sample_w_rep)

    # for db_engine, db_name, sample_w_rep in itertools.product(['mssql'], ['ssb', 'tpch', 'imdb'], [False]):
    #     prepare_data_on_dataset(db_engine=db_engine,
    #                             db_name=db_name, sample_with_replacement=sample_w_rep)
