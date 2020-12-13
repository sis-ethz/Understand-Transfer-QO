from pymongo import *
import pandas as pd
import numpy as np

client = MongoClient('localhost', 27017)
db = client["tpch"]
db = client.tpch

print(f'db {db}')

columns = {
    'nation': ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'],
    'region': ['r_regionkey', 'r_name', 'r_comment'],
    'supplier': ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal',
                 's_comment'],
    'customer': ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal',
                 'c_mktsegment', 'c_comment'],
    'part': ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container',
             'p_retailprice', 'p_comment'],
    'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'],
    'orders': ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate',
               'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'],
    'lineitem': ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
                 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus',
                 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode',
                 'l_comment']
}


# ===============================
for t in columns.keys():
    table = db[t]

    x = table.delete_many({})
    print(f"deleted {x.deleted_count}")

    data = pd.read_csv(f'tables/{t}.tbl', sep='|', header=None)
    # print(data)

    data = data.drop(data.columns[-1], axis=1)
    # print(data)
    data.columns = columns[t]

    # Set up a dictionary before import
    data = [x[1].to_dict() for x in data.iterrows()]
    # Insert dictionary data on table
    table.insert_many(data)

    print("loaded ", table.count(), f" lines for {t}")
# ===============================
