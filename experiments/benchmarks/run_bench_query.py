from core.DB_connector import *
import re
import os
import json
import tqdm
from datetime import datetime

class Couchbase_Query:

    def __init__(self, query_str):
        self.projection, self.from_tables, self.join_predicates, self.join_clause, self.predicates, self.group_by, self.order_by \
            = self._parse_query(query_str)

    def _parse_query(self, query_str):
        self.original_query = query_str
        # SELECT self.projection
        # FROM self.from_tables ON self.join_predicate
        # WHERE self.predicates
        # GROUP BY self.group_by
        # ORDER BY self.order_by;
        query_str = query_str.replace('\n', ' ').replace('\t', '')
        projection = re.search('SELECT (.*) FROM', query_str).group(1).replace('\n', '')

        from_tables = []
        join_predicates = []
        join_clause = re.search('FROM (.*) WHERE', query_str).group(1).replace('\n', '')
        join_arr = join_clause.split('INNER JOIN')
        for table in join_arr:
            if 'ON' not in table:
                from_tables.append(table.replace(' ', ''))
            else:
                table_and_pred = table.split(' ON ')
                from_tables.append(table_and_pred[0].replace(' ', ''))
                join_predicates.append(table_and_pred[-1])

        group_by, order_by = None, None
        
        if 'GROUP BY' in query_str:
            predicates = re.search('WHERE (.*) GROUP BY', query_str).group(1)
            if 'ORDER BY' in query_str:
                group_by = re.search('GROUP BY (.*) ORDER BY', query_str).group(1)
                order_by = re.search('ORDER BY (.*);', query_str).group(1)
            else:
                group_by = re.search('GROUP BY (.*);', query_str).group(1)
        elif 'ORDER BY' in query_str:
            predicates = re.search('WHERE (.*) ORDER BY', query_str).group(1)
            order_by = re.search('ORDER BY (.*);', query_str).group(1)
        else:
            predicates = re.search('WHERE (.*);', query_str).group(1)

        predicates = predicates.split('AND')
        return projection, from_tables, join_predicates, join_clause, predicates, group_by, order_by
    
    def formulate_sql(self, sql_type):
        
        
        
        assert sql_type in ['original', 'push_down', 'push_down_create_table', 'push_down_knowledge'], \
            "SQL type for couchbase is not supported yet"
        
        sql = ''

        if sql_type == 'original':
            return self.original_query
        elif sql_type == 'push_down':
            return self._formulate_push_down_query()
        elif sql_type == 'push_down_create_table':
            pass
        elif sql_type == 'push_down_knowledge':
            pass
            
        return sql
    
    def _replace_table_with_cte(self, clause, from_tables):
            for idx, t in enumerate(from_tables):
                clause = clause.replace(t, f'cte{idx}')
            return clause

    def _formulate_push_down_query(self):
        sql = ''
        cte_clauses = []
        cte_predicates = []
        top_predicates = []
        for table in self.from_tables:
            cte_clauses.append(f'SELECT * FROM {table} ')
            cte_predicates.append([])
        
        for predicate in self.predicates:
            from_table = predicate.split('.')[0]
            for idx, t in enumerate(self.from_tables):
                if t in from_table:
                    cte_predicates[idx].append(predicate)
                    break
        
        for idx, table in enumerate(self.from_tables):
            if len(cte_predicates[idx]) > 0:
                cte_clauses[idx] += ' WHERE ' + ' AND '.join(cte_predicates[idx])
        
        sql = f'WITH '
        for idx, c in enumerate(cte_clauses): 
            cte_clauses[idx] = f"cte{idx} AS ( {c} )"
        
        sql += ' ,\n '.join(cte_clauses)
        # ============== Finish predicate push down ===========
        sql += f'\nSELECT {self._replace_table_with_cte(self.projection, self.from_tables)} FROM  '

        # join_clauses = {}
        # for idx, t in enumerate(self.from_tables):
        #     if idx == 0:
        #         sql += f' cte{idx} INNER JOIN '
        join_clause = self.join_clause
        for idx, t in enumerate(self.from_tables):
            join_clause = join_clause.replace(t, f'cte{idx}')
        sql += join_clause


        if self.group_by:
            sql += '\nGROUP BY ' + replace_table_with_cte(self.group_by, self.from_tables)
        
        if self.order_by:
            sql += '\nORDER BY ' + replace_table_with_cte(self.order_by, self.from_tables)

        sql += ';'
        return sql

    def _formulate_our_query(self):
        sql = ''
        cte_clauses = []
        cte_predicates = []
        top_predicates = []
        
        for table in self.from_tables:
            cte_clauses.append(f'SELECT * FROM {table} ')
            cte_predicates.append([])
        
        for predicate in self.predicates:
            from_table = predicate.split('.')[0]
            for idx, t in enumerate(self.from_tables):
                if t in from_table:
                    cte_predicates[idx].append(predicate)
                    break
        
        for idx, table in enumerate(self.from_tables):
            if len(cte_predicates[idx]) > 0:
                cte_clauses[idx] += ' WHERE ' + ' AND '.join(cte_predicates[idx])
        
        sql = f'WITH '
        for idx, c in enumerate(cte_clauses): 
            cte_clauses[idx] = f"cte{idx} AS ( {c} )"
        
        sql += ' ,\n '.join(cte_clauses)
        # ============== Finish predicate push down ===========
        sql += f'\nSELECT {self._replace_table_with_cte(self.projection, self.from_tables)} FROM  '

        # join_clauses = {}
        # for idx, t in enumerate(self.from_tables):
        #     if idx == 0:
        #         sql += f' cte{idx} INNER JOIN '
        join_clause = self.join_clause
        for idx, t in enumerate(self.from_tables):
            join_clause = join_clause.replace(t, f'cte{idx}')
        sql += join_clause


        if self.group_by:
            sql += '\nGROUP BY ' + replace_table_with_cte(self.group_by, self.from_tables)
        
        if self.order_by:
            sql += '\nORDER BY ' + replace_table_with_cte(self.order_by, self.from_tables)

        sql += ';'
        return sql

    def _load_models(self):
        # apply this after the join is read
        pass

class benchmarker():
    
    def __init__(self, system_flag, db_name='ssb', query_dir='query/ssb/original/'):
        self.ssb_query_files = ['01.sql', '02.sql', '03.sql', '04.sql', '05.sql', '06.sql', '07.sql', '08.sql', '09.sql', '10.sql', '11.sql', '12.sql', '13.sql']
        self.tpch_query_files = [ '03.sql', '05.sql', '09.sql', '10.sql', '11.sql', '12.sql', '14.sql', '16.sql','18.sql','19.sql']
        self.imdb_query_files = [ '1.sql', '10.sql', '20.sql', '30.sql', '40.sql', '50.sql', '60.sql', '70.sql']

        # ===============================
        # self.tpch_query_files = [ '11.sql', '12.sql', '14.sql', '16.sql','18.sql','19.sql']

        # ===============================

        self.system_flag = system_flag
        self._read_queries(db_name, query_dir)
        self.db = Couchbase_Connector(execution_pass=False)
        self.db_name = db_name
        self.query_dir = query_dir

    def _read_queries(self, db_name, query_dir):
        assert db_name in query_dir
        files = []
        for f in os.listdir(query_dir):
            if '.sql' in f:
                files.append(f)
        
        if db_name == 'tpch':
            files = self.tpch_query_files
        elif db_name == 'ssb':
            files = self.ssb_query_files
        elif db_name == 'job_light':
            files = self.imdb_query_files
        
        # files = [ '30.sql', '40.sql', '50.sql', '60.sql', '70.sql']
        names = [db_name + '_' + n for n in files]
        queries = []
        for q_file in files:
            with open(os.path.join(query_dir, q_file), 'r') as f:
                q = ''
                line = f.readline()
                while line:
                    if 'statistics' in line.lower() and self.system_flag == 'original':
                        pass # skip UPDATE STATISTICS for original version
                    elif not (len(line) >= 2 and '--' == line[0:2]) and not (line == '\n'):
                        q += line
                    line = f.readline()
                if 'index' in query_dir:
                    assert 'CREATE INDEX' in q and 'DROP INDEX' in q, f"No index creation and drop in{q_file}"
                queries.append(q)
                
        self.query_names, self.query_strs = names, queries

    def run_all_queries(self, save_dir=None, repeat_num=1):
        print(f"Running queries from {self.query_dir}")
        costs = {}
        if not save_dir:
            save_dir = self.query_dir
        # with open(os.path.join(save_dir, f'run_cost_{self.system_flag}.txt'), 'a') as fp:
        #     fp.write(f"==================================\n")
        with open(os.path.join(save_dir, f'run_cost_{self.system_flag}.txt'), 'a') as fp:
                fp.write(f"\nRunning queries with repeat = {repeat_num} at {str(datetime.now())}\n")
        for name, q in zip(self.query_names, self.query_strs): 
            print(f"Executing query: {name}")

            with open(os.path.join(save_dir, f'run_cost_{self.system_flag}.txt'), 'a') as fp:
                fp.write(f"==================================\n")

            queries = q.replace('\n', ' ').split(';')

            create_queries = []
            drop_queries = []
            main_query = ''

            for sql in queries:
                if 'create' in sql.lower():
                    create_queries.append(sql)
                elif 'drop' in sql.lower() or 'delete' in sql.lower():
                    drop_queries.append(sql)
                elif 'select' in sql.lower():
                    main_query = sql

            for idx in range(repeat_num):
                if idx == 0:
                    self.db.execute(';'.join(create_queries) + ';')
                res_dict = self.db.execute(main_query)
                costs[name] = res_dict['execution_cost']
                with open(os.path.join(save_dir, f'run_cost_{self.system_flag}.txt'), 'a') as fp:
                    log = f"Query {name} ({idx} / {repeat_num}) : {'%.2f' % costs[name]}, result size: {res_dict['estimated_result_size']}\n"
                    fp.write(log)
                    print(log)
                if idx == repeat_num - 1 or costs[name] < 0:
                    self.db.execute(';'.join(drop_queries) + ';')
                    break

            with open(os.path.join(save_dir, f'run_cost_{self.system_flag}.txt'), 'a') as fp:
                fp.write(f"==================================\n")

        return costs


    def transform_queries(self, save_dir, to_type):
        for name, q in zip(self.query_names, self.query_strs): 
            name = name.replace(self.db_name+'_', '')
            couchbase_q = Couchbase_Query(q)
            q = couchbase_q.formulate_sql(sql_type=to_type)
            with open(os.path.join(save_dir, name), 'w') as g:
                g.write(q)
            


if __name__ == "__main__":

    # query = """SELECT SUM(ssb_lineorder.lo_extendedprice*ssb_lineorder.lo_discount) AS revenue
    #     FROM ssb_lineorder INNER JOIN ssb_ddate ON ssb_lineorder.lo_orderdate = ssb_ddate.d_datekey
    #     WHERE ssb_ddate.d_yearmonthnum = 199401
    #     AND ssb_lineorder.lo_discount BETWEEN 4 AND 6
    #     AND ssb_lineorder.lo_quantity BETWEEN 26 AND 35;
    #     """
    # q = Couchbase_Query(query)
    # print(q.formulate_sql(sql_type='push_down'))

    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    parser.add_argument('--system', dest='system', action='store',
                        default='original')

    parser.add_argument('--data-set', dest='data_set', action='store',
                        default='ssb')

    parser.add_argument('--query-set', dest='query_set', action='store',
                        default='create_index_our_knowledge')
    
    parser.add_argument('--repeat', dest='repeat_times', action='store',
                        default='1')

    args = parser.parse_args()

    runner = benchmarker(system_flag=args.system, db_name=args.data_set, query_dir=f'query/{args.data_set}/{args.query_set}/')
    runner.run_all_queries(repeat_num=int(args.repeat_times))

    # # ===============================================
    # # Original Couchbase + Original query
    # system = 'original'
    # runner = benchmarker(system_flag=system, db_name='ssb', query_dir='query/ssb/original/')
    # runner.run_all_queries()
    # runner = benchmarker(system_flag=system, db_name='tpch', query_dir='query/tpch/original/')
    # runner.run_all_queries() 
    # # ==============================================

    # ===============================================
    # Original Couchbase + Create index
    # system = 'original'
    # runner = benchmarker(system_flag=system, db_name='ssb', query_dir='query/ssb/create_index/')
    # runner.run_all_queries()
    # runner = benchmarker(system_flag=system, db_name='tpch', query_dir='query/tpch/create_index/')
    # runner.run_all_queries() 
    # ==============================================

    # ===============================================
    # Original Couchbase + Create index + Our knowledge
    # system = 'original'
    # runner = benchmarker(system_flag='orginal', db_name='ssb', query_dir='query/ssb/create_index_our_knowledge/')
    # runner.run_all_queries() 
    # runner = benchmarker(system_flag='orginal', db_name='tpch', query_dir='query/tpch/create_index_our_knowledge/')
    # runner.run_all_queries() 
    # ==============================================


    # ===============================================
    # Preview Couchbase + Original query
    # runner = benchmarker(system_flag='preview', db_name='ssb', query_dir='query/ssb/original/')
    # runner.run_all_queries() 

    # runner = benchmarker(system_flag='preview', db_name='tpch', query_dir='query/tpch/original/')
    # runner.run_all_queries() 
    # ==============================================

    # ===============================================
    # Original Couchbase + Create index
    # system = 'original'
    # runner = benchmarker(system_flag=system, db_name='ssb', query_dir='query/ssb/create_index/')
    # runner.run_all_queries()
    # runner = benchmarker(system_flag=system, db_name='tpch', query_dir='query/tpch/create_index/')
    # runner.run_all_queries() 
    # ==============================================