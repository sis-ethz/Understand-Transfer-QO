# Database Connectors
# import pyodbc
from cardinality_estimation_quality.cardinality_estimation_quality import *
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from collections import *


class Postgres_Connector:
    def __init__(self, server='localhost', username='postgres', password='postgres', db_name=None):
        self.server = server
        self.username = username
        self.password = password
        self.db_name = db_name

        if db_name:
            self.db_url = f'host={server} port=5432 user={username} dbname={db_name} password={password}'
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


class Mongo_Connector:
    pass


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
    mssql = Mssql_Connector(db_name='imdb')

    q = """
    WITH prev_result_view AS (select TOP 43568 * from lineitem ORDER BY NEWID())
 select * from prev_result_view, part WITH (INDEX(part_partkey))  where part.p_partkey = prev_result_view.l_partkey and part.p_partkey > 199341 OPTION (MERGE JOIN);
    """
    plan = mssql.explain(q)

    print(plan)
