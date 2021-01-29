
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

table_indexed_keys = {
    'part': 'p_partkey',
    'supplier': 's_suppkey',
    'customer': 'c_custkey',
    'date': 'd_datekey',
    'lineorder': 'lo_orderkey'
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


def create_index_on_each_attr(table_list, table_attrs):
    cmds = []

    create_index_cmd = f'/opt/couchbase/bin/cbq -c 127.0.0.1 -u couchbase -p couchbase '

    for t in table_list:
        if t == 'date':
            table_name = 'ddate'
        else:
            table_name = t

        for attr in table_attrs[t]:
            # cmds.append(f"DROP INDEX `ssb_{table_name}`.ssb_{table_name}_{attr};")
            cmds.append(f'CREATE INDEX ssb_{table_name}_{attr} ON `ssb_{table_name}` ( {attr} );')

    print('\n'.join(cmds))


for t in table_list:
    if t == 'date':
        table_name = 'ddate'
    else:
        table_name = t
    
    bkt_del_cmd = f'/opt/couchbase/bin/couchbase-cli bucket-delete -c 127.0.0.1 -u couchbase -p couchbase --bucket ssb_{table_name}'
    bkt_create_cmd = f'/opt/couchbase/bin/couchbase-cli bucket-create -c 127.0.0.1 -u couchbase -p couchbase --bucket ssb_{table_name} --bucket-type couchbase --bucket-ramsize 1000'
    import_cmd = f"/opt/couchbase/bin/cbimport csv -c 127.0.0.1 -u couchbase -p couchbase -b ssb_{table_name} -d file:///mnt/interpretable-cost-model/datasets/ssb/tables/{table_name}_couchbase.tbl --infer-types -g key::%{table_attrs[t][0]}%"
    create_index_cmd = f''

    # print("del buckt cmd: ", bkt_del_cmd)
    # os.system(bkt_del_cmd)

    # print("create buckt cmd: ", bkt_create_cmd)
    # os.system(bkt_create_cmd)
    # print("import cmd: ", import_cmd)
    # os.system(import_cmd)

    # for attr in table_attrs[t]:
    #     # cmds.append(f"DROP INDEX `ssb_{table_name}`.ssb_{table_name}_{attr};")
    #     print(f'DROP INDEX `ssb_{table_name}`.ssb_{table_name}_{attr};')
    #     print(f'DROP INDEX `temp_ssb_{table_name}`.temp_ssb_{table_name}_{attr};')

        # if 'key' not in  attr:
        # print(f'CREATE INDEX ssb_{table_name}_{attr} ON `ssb_{table_name}` ({attr} );')

    # print(f'CREATE INDEX ssb_{table_name}_all ON `ssb_{table_name}` ( {",".join(table_attrs[t])} );')
    print(f'DROP INDEX `ssb_{table_name}`.ssb_{table_name}_all;')
    # print(f'CREATE INDEX ssb_{table_name}_key ON `ssb_{table_name}` ( {table_indexed_keys[t]} );')

    
    # print("create index:", f'CREATE INDEX temp_ssb_part_{table_indexed_keys[t]} ON `ssb_{table_name}` ({table_indexed_keys[t]});')

    # print(f'CREATE PRIMARY INDEX ssb_{table_name}_primary ON `ssb_{table_name}`;')
    
    # print(" ", f"UPDATE STATISTICS FOR `ssb_{table_name}` ({','.join(table_attrs[t])});")

    # select * from ssb_customer c JOIN ssb_lineorder lo USE HASH(BUILD) ON c.c_custkey = lo.lo_custkey;