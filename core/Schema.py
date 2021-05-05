import numpy as np


class Schema:
    def __init__(self):
        self.primary_keys = {}
        self.join_keys = {}
        self.table_features = {}
        self.join_graph = {}
        self.indexes = {}

    def random_indexed_attr_value(self, table):
        key_range = self.table_features[table]['key_range']
        return np.random.randint(key_range[-1] - key_range[0]) + key_range[0]

    def random_non_indexed_attr_value(self, table):
        _, attr_range = self.non_indexed_attr[table]
        return np.random.randint(attr_range[-1] - attr_range[0]) + attr_range[0]


class IMDB_lite_schema(Schema):

    def __init__(self,):
        super(IMDB_lite_schema, self).__init__()

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

        self.non_indexed_attr = {
            # table: one non_indexed attr, and its range
            'movie_info_idx': ('movie_id', [2, 2525793]),
            'movie_info': ('movie_id', [1, 2526430]),
            'movie_companies': ('movie_id', [2, 2525745]),
            'movie_keyword': ('movie_id', [2, 2525971]),
            'cast_info': ('movie_id', [1, 2525975]),
            'title': ('production_year', [1880, 2019])
        }

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

        # the database used by sampler

        self.join_graph = {
            # constructed by base table + left table list
            'movie_info_idx': ['title'],
            'movie_info': ['title'],
            'movie_companies': ['title'],
            'movie_keyword': ['title'],
            'cast_info': ['title'],
            'title': ['movie_companies', 'movie_info_idx', 'movie_info', 'movie_keyword', 'cast_info']
        }

        self.indexes = {
            # indexed attr: index size on postgres
            'movie_info_idx': {
                'id': 31014912
            },
            'movie_info': {
                'id': 482312192
            },
            'movie_companies': {
                'id':  58621952
            },
            'movie_keyword': {
                'id':  101629952
            },
            'cast_info': {
                'id':  814112768
            },
            'title': {
                'id':  73875456
            },
        }


class SSB_Schema(Schema):

    def __init__(self,):
        super(SSB_Schema, self).__init__()

        self.tables = [
            'part', 'lineitem', 'supplier', 'orders', 'ddate']

        self.primary_keys = {
            'part': 'p_partkey',
            'supplier': 's_suppkey',
            'ddate': 'd_datekey',
            'customer': 'c_custkey',
            'lineorder': "",
        }  # table: primary key

        self.join_keys = {
            ('lineorder', 'customer'): ('lo_custkey', 'c_custkey'),
            ('customer', 'lineorder'): ('c_custkey', 'lo_custkey'),

            ('supplier', 'lineorder'): ('s_suppkey', 'lo_suppkey'),
            ('lineorder', 'supplier'): ('lo_suppkey', 's_suppkey'),

            ('part', 'lineorder'): ('p_partkey', 'lo_partkey'),
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

        self.join_graph = {
            # constructed by base table + left table list
            'part': ['lineorder'],
            'customer': ['lineorder'],
            'ddate': ['lineorder'],
            'supplier': ['lineorder'],

            # full graphs: 
            'lineorder': ['part', 'customer', 'ddate', 'supplier'],
        }

        self.non_indexed_attr = {
            # table: one non_indexed attr, and its range
            'part': ('p_size', [1, 50]),
            'customer': ('c_name', ['Customer#000000001', 'Customer#000030000']),
            'ddate': ('d_daynuminyear', [1, 366]),
            'supplier': ('s_name', ['Supplier#000000001', 'Supplier#000002000']),
            'lineorder': ('lo_custkey', [1,  30000])
        }

        self.indexes = {
            'part': {
                'p_partkey': 4513792
            },
            'supplier': {
                's_suppkey': 65536
            },
            'ddate': {
                'd_datekey': 73728
            },
            'customer': {
                'c_custkey': 688128
            },
            'lineorder': {
                '':  -1,
                # 'lo_linenumber'
            },
        }

    def random_non_indexed_attr_value(self, table):
        if table not in ['supplier', 'customer']:
            return super().random_non_indexed_attr_value(table)
        elif table == 'supplier':
            attr_range = [1, 30000]
            rand_attr = str(np.random.randint(
                attr_range[-1] - attr_range[0]) + attr_range[0])
            return f'\'Supplier#{rand_attr.zfill(9)}\''
        else:
            attr_range = [1, 2000]
            rand_attr = str(np.random.randint(
                attr_range[-1] - attr_range[0]) + attr_range[0])
            return f'\'Customer#{rand_attr.zfill(9)}\''


class TPCH_Schema(Schema):
    def __init__(self,):
        super(TPCH_Schema, self).__init__()
        self.tables = ['part', 'lineitem',
                       'supplier', 'orders', 'partsupp', 'customer',
                       'nation', 'region']

        self.primary_keys = {
            # table: primary key
            'part': 'p_partkey',
            'supplier': 's_suppkey',
            'orders': 'o_orderkey',
            'customer': 'c_custkey',
            'partsupp': '',
            'lineitem': '',
            'nation': 'n_nationkey',
            'region': 'r_regionkey'
        }

        self.join_keys = {
            # (table, table) : (key1, key2)
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

            # ('lineitem', 'partsupp'): ('l_suppkey', 'ps_suppkey'),
            # ('partsupp', 'lineitem'): ('ps_suppkey', 'l_suppkey')

            ('nation', 'region') : ('n_regionkey', 'r_regionkey'),
            ('region', 'nation') : ('r_regionkey', 'n_regionkey')

        }

        self.table_features = {
            'part': {
                'table_size': 200000,
                'key_range': [1, 200000]
            },

            'orders': {
                'table_size': 1500000,
                'key_range': [1, 6000000]
                # 'table_size': 1500000,
                # 'key_range': [1,  149999]
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
                'key_range': [-1, -1]
            },
            'nation': {
                'table_size': 25,
                'key_range': [1, 25]
            },
            'region': {
                'table_size': 5,
                'key_range': [1, 5]
            },

            'lineitem': {
                'table_size': 6000003,
                'key_range': [-1, -1]
            }
        }  # table features

        self.left_size_ratio_threshold = 0.5

        self.join_graph = {
            # # constructed by base table + left table list
            # 'part': ['lineitem', 'partsupp'],
            # 'orders': ['lineitem', 'customer'],
            # 'supplier': ['lineitem', 'partsupp'],
            # 'customer': ['orders'],
            # 'partsupp': ['lineitem', 'part',  'supplier'],
            # 'lineitem': ['orders', 'part', 'partsupp'],
            # 'region': ['nation'],
            # 'nation': ['region'],

            # constructed by base table + left table list
            'supplier': ['partsupp', 'lineitem'],
            'customer': ['orders'],
            'region': ['nation'],
            'nation': ['region'],
            'part': ['lineitem', 'partsupp'],
            'orders': ['lineitem', 'customer'],
            'partsupp': ['lineitem', 'part',  'supplier'],
            'lineitem': ['orders', 'part', 'partsupp'],
        }

        self.non_indexed_attr = {
            # table: one non_indexed attr, and its range
            'part': ('p_size', [1, 50]),
            'orders': ('o_custkey', [1, 149999]),
            'supplier': ('s_nationkey', [0, 24]),
            'customer': ('c_nationkey', [0, 24]),
            'partsupp': ('ps_suppkey', [1, 10000]),
            'lineitem': ('l_partkey', [1, 200000]),
            'nation': ('n_regionkey', [1, 5]), 
            'region': ('', [-1, 0])
        }

        self.indexes = {
            'part': {
                'p_partkey': 4513792
            },
            'orders': {
                'o_orderkey': 33718272
            },
            'supplier': {
                's_suppkey': 245760
            },
            'customer': {
                'c_custkey':  3391488
            },
            'partsupp': {
                '': -1
            },
            'lineitem': {
                '': -1
            },
            'nation': {
                'n_nationkey': -1
            }, 
            'region': {
                'r_regionkey': -1
            }
        }

        self.index_size = {

        }


class TPCH_100_Schema(TPCH_Schema):

    def __init__(self,):
        super(TPCH_100_Schema, self).__init__()

        self.table_features = {
            'part': {
                'table_size': 20000000,
                'key_range': [1, 20000000]
            },

            'orders': {
                'table_size': 150000000,
                'key_range': [1, 600000000]
                # 'table_size': 1500000,
                # 'key_range': [1,  149999]
            },
            'supplier': {
                'table_size': 1000000,
                'key_range': [1, 1000000]
            },
            'customer': {
                'table_size': 15000000,
                'key_range': [1, 15000000]
            },
            'partsupp': {
                'table_size': 80000000,
                'key_range': [-1, -1]
            },
            'nation': {
                'table_size': 25,
                'key_range': [1, 25]
            },
            'region': {
                'table_size': 5,
                'key_range': [1, 5]
            },

            'lineitem': {
                'table_size': 600037902,
                'key_range': [-1, -1]
            }
        }  # table features

        self.non_indexed_attr = {
                # table: one non_indexed attr, and its range
                'part': ('p_size', [1, 50]),
                'orders': ('o_custkey', [1, 14999999]),
                'supplier': ('s_nationkey', [0, 24]),
                'customer': ('c_nationkey', [0, 24]),
                'partsupp': ('ps_suppkey', [1, 1000000]),
                'lineitem': ('l_partkey', [1, 20000000]),
                'nation': ('n_regionkey', [1, 5]), 
                'region': ('', [-1, 0])
        }

        self.indexes = {
            'part': {
                'p_partkey': 451379200
            },
            'orders': {
                'o_orderkey': 3371827200
            },
            'supplier': {
                's_suppkey': 245760000
            },
            'customer': {
                'c_custkey':  339148800
            },
            'partsupp': {
                '': -1
            },
            'lineitem': {
                '': -1
            },
            'nation': {
                'n_nationkey': -1
            }, 
            'region': {
                'r_regionkey': -1
            }
        }

class TPCH_10_Schema(TPCH_Schema):

    def __init__(self,):
        super(TPCH_10_Schema, self).__init__()

        self.table_features = {
            'part': {
                'table_size': 2000000,
                'key_range': [1, 2000000]
            },

            'orders': {
                'table_size': 15000000,
                'key_range': [1, 60000000]
                # 'table_size': 1500000,
                # 'key_range': [1,  149999]
            },
            'supplier': {
                'table_size': 100000,
                'key_range': [1, 100000]
            },
            'customer': {
                'table_size': 1500000,
                'key_range': [1, 1500000]
            },
            'partsupp': {
                'table_size': 8000000,
                'key_range': [-1, -1]
            },
            'nation': {
                'table_size': 25,
                'key_range': [1, 25]
            },
            'region': {
                'table_size': 5,
                'key_range': [1, 5]
            },

            'lineitem': {
                'table_size': 59986052,
                'key_range': [-1, -1]
            }
        }  # table features

        self.non_indexed_attr = {
                # table: one non_indexed attr, and its range
                'part': ('p_size', [1, 50]),
                'orders': ('o_custkey', [1, 1499999]),
                'supplier': ('s_nationkey', [0, 24]),
                'customer': ('c_nationkey', [0, 24]),
                'partsupp': ('ps_suppkey', [1, 100000]),
                'lineitem': ('l_partkey', [1, 2000000]),
                'nation': ('n_regionkey', [1, 5]), 
                'region': ('', [-1, 0])
        }

        self.indexes = {
            'part': {
                'p_partkey': 45137920
            },
            'orders': {
                'o_orderkey': 337182720
            },
            'supplier': {
                's_suppkey': 24576000
            },
            'customer': {
                'c_custkey':  33914880
            },
            'partsupp': {
                '': -1
            },
            'lineitem': {
                '': -1
            },
            'nation': {
                'n_nationkey': -1
            }, 
            'region': {
                'r_regionkey': -1
            }
        }