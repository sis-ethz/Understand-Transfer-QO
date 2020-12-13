
import os
from os import path

table_list = ['part', 'customer', 'date', 'lineorder', 'supplier']
# table_list = ['customer', 'date', 'lineorder', 'supplier']
table_attrs = {
    'part': ['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type', 'p_size', 'p_container'],
    'supplier': ['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region', 's_phone'],
    'customer': ['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region', 'c_phone', 'c_mktsegment'],
    'date': ['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum', 'd_yearmonth', 'd_daynuminweek',
             'd_daynuminmonth', 'd_daynuminyear', 'd_monthnuminyear', 'd_weeknuminyear', 'd_sellingseason',
             'd_lastdayinweekfl', 'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl'],
    'lineorder': ['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey', 'lo_orderdate', 'lo_orderpriority',
                  'lo_shippriority', 'lo_quantity', 'lo_extendedprice', 'lo_ordertotalprice', 'lo_discount', 'lo_revenue',
                  'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmode']
}

# for t in ['date']:

#     # if not path.exists(t + '_couchbase.tbl'):
#     with open(t + '.tbl', 'r') as f, open(t + '_couchbase.tbl', 'w') as g:
#         g.write(','.join(table_attrs[t]) + '\n')
#         print("tbl: ", ','.join(table_attrs[t]))
#         line = f.readline()
#         while line:
#             # print("Read line:", line)
#             line = line.replace("\n", '').replace(
#                 ",", ' ').strip('|').replace("|", ',')
#             g.write(line + '\n')
#             line = f.readline()

for t in table_list:
    if t == 'date':
        table_name = 'ddate'
    else:
        table_name = t
    
    bkt_del_cmd = f'couchbase-cli bucket-delete -c 127.0.0.1 -u couchbase -p couchbase --bucket ssb_{table_name}'
    bkt_create_cmd = f'couchbase-cli bucket-create -c 127.0.0.1 -u couchbase -p couchbase --bucket ssb_{table_name} --bucket-type couchbase --bucket-ramsize 1000'
    import_cmd = f"cbimport csv -c 127.0.0.1 -u couchbase -p couchbase -b ssb_{table_name} -d file:///mnt/interpretable-cost-model/datasets/ssb/tables/{table_name}_couchbase.tbl --infer-types -g key::%{table_attrs[t][0]}%"
    create_index_cmd = f''

    # print("del buckt cmd: ", bkt_del_cmd)
    # os.system(bkt_del_cmd)

    # print("create buckt cmd: ", bkt_create_cmd)
    # os.system(bkt_create_cmd)
    # print("import cmd: ", import_cmd)
    # os.system(import_cmd)

    print("create index:", f'CREATE PRIMARY INDEX `ssb_{table_name}_primary` ON `ssb_{table_name}`')
