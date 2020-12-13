from pymongo import *
import pandas as pd
import numpy as np

client = MongoClient('localhost', 27017)
db = client["ssb"]
db = client.ssb

print(f'db {db}')

columns = {
    'part': ['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type',
             'p_size', 'p_container'],
    'supplier': ['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region',
                 's_phone'],
    'customer': ['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region',
                 'c_phone', 'c_mktsegment'],
    'ddate': ['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum',
              'd_yearmonth', 'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear',
              'd_monthnuminyear', 'd_weeknuminyear', 'd_sellingseason', 'd_lastdayinweekfl',
              'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl'],
    'lineorder': ['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey',
                  'lo_orderdate', 'lo_orderpriority', 'lo_shippriority', 'lo_quantity',
                  'lo_extendedprice', 'lo_ordertotalprice', 'lo_discount', 'lo_revenue',
                  'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmode']
}


# ===============================
# for t in columns.keys():
#     table = db[t]

#     x = table.delete_many({})
#     print(f"deleted {x.deleted_count}")

#     if t == 'ddate':
#         data = pd.read_csv(f'tables/date.tbl', sep='|', header=None)
#     else:
#         data = pd.read_csv(f'tables/{t}.tbl', sep='|', header=None)

#     print(data)

#     data = data.drop(data.columns[-1], axis=1)
#     print(data)
#     data.columns = columns[t]

#     # Set up a dictionary before import
#     data = [x[1].to_dict() for x in data.iterrows()]
#     # Insert dictionary data on table
#     table.insert_many(data)

#     print("loaded ", table.count(), f" lines for {t}")
# ===============================

pipeline = [{
    "$lookup": {
        "from": "customer",
        "localField": "lo_custkey",
        "foreignField": "c_custkey",
        "as": "join_result"
    }
}]


res = db.command('aggregate', 'lineorder', pipeline=pipeline, explain=True)

print(res)
