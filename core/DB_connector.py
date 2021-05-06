# Database Connectors

# ====================
# MSSQL
# import pyodbc
# ====================

# ====================
# Postgres
from core.cardinality_estimation_quality.cardinality_estimation_quality import *
# ====================

# ====================
## Couchbase 
## load couchbase stuffs
# import couchbase.search as FT
# import couchbase.subdocument as SD
# # import jwt  # from PyJWT
# from couchbase.cluster import Cluster, ClusterOptions, PasswordAuthenticator, ClusterTimeoutOptions
# from couchbase.exceptions import *
# from couchbase.search import SearchOptions
# from couchbase.exceptions import TimeoutException
# ====================

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import *
import time
import json
from datetime import timedelta

class Postgres_Connector:
    def __init__(self, server='localhost', username='postgres', password='postgres', db_name=None):
        self.server = server
        self.username = username
        self.password = password
        self.db_name = db_name

        if db_name:
            self.db_url = f"host={server} port=5432 user={username} dbname={db_name} password={password} options='-c statement_timeout={12000000}' "
            self.init_db(db_name)

    def init_db(self, db_name):
        db = self.db_url.format(db_name)
        PG = Postgres(db)
        self.db = PG
        return PG

    def disable_parallel(self):
        self.execute(
            'LOAD \'pg_hint_plan\';SET max_parallel_workers_per_gather=0;SET max_parallel_workers=0;', set_env=True)

    def explain(self, query, timeout=0):
        q = QueryResult(None)
        q.query = query
        q.explain(self.db, execute=False, timeout=timeout)
        return q

    def execute(self, query, set_env=False):
        start_time = time.time()
        res = self.db.execute(query, set_env=set_env)
        end_time = time.time()
        q = {
            'execution_cost': end_time - start_time,
            'estimated_result_size': len(res)
        }
        # print(res)
        return q


class Couchbase_Connector:
    def __init__(self, server='127.0.0.1:8091', username='couchbase', password='couchbase', db_name=None, execution_pass=False):
        self.server = server
        self.username = username
        self.password = password
        self.db_name = db_name
        # timeout_options = ClusterTimeoutOptions(config_total_timeout=timedelta(hours=5), kv_timeout=timedelta(hours=1), query_timeout=timedelta(hours=1), views_timeout=timedelta(hours=1))
        timeout_options = ClusterTimeoutOptions(config_total_timeout=timedelta(minutes=20), kv_timeout=timedelta(minutes=20), query_timeout=timedelta(minutes=20))
        options = ClusterOptions(PasswordAuthenticator(username, password), timeout_options=timeout_options)
        # print("Set timeout: 10min")
        self.cluster = Cluster(f'couchbase://{server}', options)
        self.execution_pass = execution_pass
        if execution_pass:
            print("[Warning]: each execution of this Couchbase node is passed for futher execution. Remember to re-execute the queries!")
            
        # self.cluster.authenticate(PasswordAuthenticator(username, password))

    def refresh(self, timeout=None):
        if timeout is None:
            # Default timeout: 20min
            timeout = 20
        timeout_options = ClusterTimeoutOptions(config_total_timeout=timedelta(minutes=timeout), kv_timeout=timedelta(minutes=timeout), query_timeout=timedelta(minutes=timeout))
        options = ClusterOptions(PasswordAuthenticator(self.username, self.password), timeout_options=timeout_options)
        self.cluster = Cluster(f'couchbase://{self.server}', options)


    def explain(self, query):
        if 'explain' not in query.lower():
            query = 'EXPLAIN ' + query
        result = self.cluster.query(query)
        for row in result:
            return row
        # return result[0]
        return None

    def execute(self, query):
        # reconnect using new user

        # must have at least one open bucket to submit cluster query
        # bucket = cluster.open_bucket('travel-sample')

        # Perform a N1QL Query to return document IDs from the bucket. These IDs will be
        # used to reference each document in turn, and check for extended attributes
        # corresponding to discounts.
        if 'create' in query:
            self.refresh(timeout=40)
        else:
            self.refresh()

        query = query.replace('\n', ' ').strip(' ')
        # print("Execute: ", query)

        q = {
            'execution_cost': 0,
            'estimated_result_size': 0
        }
        
        if len(query) <= 0:
            return q
        elif self.execution_pass:
            q = {
                'estimated_result_size': -3,
                'execution_cost': -3
            }
            return q
        else:
            if ';' in query:
                # print(query.split(';'))
                # exit(1)
                quries = query.split(';')

                for sql in quries:
                    assert ';' not in sql

                    print("Execute: ", sql)
                    
                    current_q = self.execute(sql)
                    if 'delete' in sql.lower() or 'drop' in sql.lower() or 'create' in sql.lower():
                        q['execution_cost'] += 0
                        # q['estimated_result_size'] = current_q['estimated_result_size']
                    elif "select" in sql.lower() :
                        q['execution_cost'] = current_q['execution_cost']
                        q['estimated_result_size'] = current_q['estimated_result_size']

                return q

            else:
                q = {}
                start = time.time()
                result = self.cluster.query(query)
                try:
                    row_cnt = 0
                    for row in result:
                        # make sure rows are materialized, rather than intermediate representation
                        # print(row)
                        row_cnt += 1
                    end = time.time()
                except TimeoutException as e:
                    row_cnt = -100
                    end = 0
                    start = 100
                    print(f"Timeout query: {query}")
                except Exception as e:
                    row_cnt = -2
                    end = 0
                    start = 2
                    print("Error: ")
                    print(e)
                    print("Query: ")
                    print(query)
                    # exit(1)
                    
                q['estimated_result_size'] = row_cnt
                q['execution_cost'] = end - start
                # print(f"return with size {row_cnt}")
                return q

    

class Mssql_Connector:
    def __init__(self, server='localhost', username='SA', password='SQLServer123', db_name=None):

        self.server = server
        self.username = username
        self.password = password
        self.db_name = db_name

        if db_name:
            cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' +
                                  server+';DATABASE='+db_name+';UID='+username+';PWD=' + password)
            cursor = cnxn.cursor()
            self.cursor = cursor

    def execute(self, query, set_env=False):

        if not set_env:
            self.cursor.execute(query)
            row = self.cursor.fetchone()
            while row:
                print(row)
                return
                row = self.cursor.fetchone()
        else:
            self.cursor.execute(query)

    def explain(self, query, show_xml=False):
        # getting the query plan with XML format

        self.execute('SET SHOWPLAN_XML ON;', set_env=True)
        self.execute(
            'ALTER DATABASE SCOPED CONFIGURATION SET BATCH_MODE_ADAPTIVE_JOINS = OFF;', set_env=True)

        self.cursor.execute(query)
        row = self.cursor.fetchone()
        raw_xml_plan = row[0]

        self.execute('SET SHOWPLAN_XML OFF;', set_env=True)
        res_feat_dict = self.xml_plan_parser(raw_xml_plan)

        if show_xml:
            print(
                f"Plan for {query}\n==================\n{raw_xml_plan}\n==================")

        return res_feat_dict

    def xml_plan_parser(self, raw_xml_plan, parse_join=True):

        def bfs(root, target_node):
            q = deque()
            q.append(root)
            while q:
                node = q.popleft()
                if node and target_node in node.tag:
                    return node
                elif not node:
                    continue
                else:
                    for child in node:
                        q.append(child)
            return None

        def bfs_join_node(root):
            # Assume that there is only one join node in the tree
            # PhysicalOp="Nested Loops"
            # PhysicalOp="Hash Match"
            # PhysicalOp = "Merge Join"

            # return the two join children

            left_child, right_child = None, None
            join_method = None

            q = deque()
            q.append(root)
            got_join = False
            while q:
                node = q.popleft()
                if not node:
                    continue
                elif "LogicalOp" in node.attrib.keys() and node.attrib["LogicalOp"] == "Inner Join":
                    join_method = node.attrib["PhysicalOp"]
                    got_join = True
                    for child in node:
                        q.append(child)
                else:
                    if got_join:
                        ret = []
                        for child in node:
                            if "RelOp" in child.tag:
                                ret.append(child)
                            q.append(child)
                        if len(ret) == 2:
                            # print(ret)
                            # print(raw_xml_plan)
                            left_child, right_child = tuple(ret)
                            break
                    else:
                        for child in node:
                            q.append(child)

            # if join_method is not None and'merge' in join_method.lower() and left_child is not None and right_child is not None:
            #     if 'Sort' in left_child.attrib["LogicalOp"]:
            #         left_child = bfs_first_RelOp(left_child)
            #     if 'Sort' in right_child.attrib["LogicalOp"]:
            #         right_child = bfs_first_RelOp(right_child)

            return left_child, right_child

        def bfs_first_RelOp(root):
            q = deque()
            q.append(root)
            while q:
                node = q.popleft()
                if node is not None and node != root and 'RelOp' in node.tag:
                    return node
                elif not node:
                    continue
                else:
                    for child in node:
                        q.append(child)
            return None

        def bfs_object_node(root):
            q = deque()
            q.append(root)
            while q:
                node = q.popleft()
                if node is not None and 'Object' in node.tag:
                    return node.attrib["Table"].replace("[", '').replace("]", '')
                elif not node:
                    continue
                else:
                    for child in node:
                        q.append(child)
            return None

        # print(raw_xml_plan)
        ret_dict = {}

        # EstimatedTotalSubtreeCost is the estimated cost of the subtree
        root = ET.fromstring(raw_xml_plan)
        # print(root.tag)

        query_plan_root = bfs(root, 'QueryPlan')
        root_op = bfs(query_plan_root, 'RelOp')
        ret_dict["total_estimated_cost"] = float(
            root_op.attrib['EstimatedTotalSubtreeCost'])
        ret_dict["estimated_result_size"] = float(
            root_op.attrib['EstimateRows'])

        # Search for join operator
        left_node, right_node = bfs_join_node(root_op)

        if left_node and right_node:
            # there is a join node in the graph
            ret_dict["left_table"] = bfs_object_node(left_node)
            ret_dict["right_table"] = bfs_object_node(right_node)
            assert ret_dict["left_table"] and ret_dict["right_table"]
            ret_dict["left_cost"] = float(
                left_node.attrib['EstimatedTotalSubtreeCost'])
            ret_dict["right_cost"] = float(
                right_node.attrib['EstimatedTotalSubtreeCost'])

        return ret_dict


if __name__ == '__main__':
    couchbase = Couchbase_Connector(db_name='ssb')
    # q = """
    #     WITH prev_result_view AS (select TOP 43568 * from lineitem ORDER BY NEWID())
    #     select * from prev_result_view, part WITH (INDEX(part_partkey))  where part.p_partkey = prev_result_view.l_partkey and part.p_partkey > 199341 OPTION (MERGE JOIN);
    #     """
    # query1 = """ WITH left_table AS (
    #                 SELECT *
    #                 FROM ssb_ddate
    #                 WHERE ssb_ddate.d_year = 1993),
    #             right_table AS (
    #                 SELECT *
    #                 FROM ssb_lineorder
    #                 WHERE ssb_lineorder.lo_discount BETWEEN 1 AND 3
    #                     AND ssb_lineorder.lo_quantity < 25)
    #             SELECT SUM(right_table.lo_extendedprice * right_table.lo_discount) AS revenue
    #             FROM left_table
    #             INNER JOIN right_table ON right_table.lo_orderdate = left_table.d_datekey;"""
    query2 =  """ 
                SELECT SUM(ssb_lineorder.lo_extendedprice * ssb_lineorder.lo_discount) AS revenue
                FROM ssb_lineorder INNER JOIN  ssb_ddate ON  ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
                WHERE ssb_ddate.d_year = 1993
                AND ssb_lineorder.lo_discount BETWEEN 1 AND 3
                AND ssb_lineorder.lo_quantity < 25;
            """

    # query = 'select * from ssb_ddate;'
    # result1 = couchbase.execute(query1)
    result2 = couchbase.execute(query2)
    # print(result1)
    print(result2)
    
