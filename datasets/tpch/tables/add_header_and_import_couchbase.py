
import os
from os import path

table_list = ['nation', 'region', 'supplier', 'customer', 'part', 'partsupp', 'orders', 'lineitem']
# table_list = ['customer', 'date', 'lineorder', 'supplier']

table_attrs = {
    'nation': ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'],
    'region': ['r_regionkey', 'r_name', 'r_comment'],
    'supplier': ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'],
    'customer': ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment'],
    'part': ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'],
    'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'],
    'orders': ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'],
    'lineitem': ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 
                    'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment']
}

table_indexed_keys = {
    'nation': 'n_nationkey',
    "region": 'r_regionkey',
    'supplier': 's_suppkey',
    'customer': 'c_custkey',
    'part': 'p_partkey',
    'partsupp': 'ps_partkey',
    'orders': 'o_orderkey',
    'lineitem': 'l_orderkey'
}

# for t in table_list:

#     # if not path.exists(t + '_couchbase.tbl'):
#     with open(t + '_fixed.tbl', 'r') as f, open(t + '_couchbase.tbl', 'w') as g:
#         g.write(','.join(table_attrs[t]) + '\n')
#         print("tbl: ", ','.join(table_attrs[t]))
#         line = f.readline()
#         while line:
#             # print("Read line:", line)
#             line = line.replace("\n", '').replace(
#                 ",", ' ').strip('|').replace("|", ',')
#             g.write(line + '\n')
#             line = f.readline()


# def create_index_on_each_attr(table_list, table_attrs):
#     cmds = []

#     create_index_cmd = f'/opt/couchbase/bin/cbq -c 127.0.0.1 -u couchbase -p couchbase '

#     for t in table_list:
#         if t == 'date':
#             table_name = 'ddate'
#         else:
#             table_name = t

#         for attr in table_attrs[t]:
#             cmds.append(f'CREATE INDEX tpch_{table_name}_{attr} ON `tpch_{table_name}` ( {attr} );')

#     print('\n'.join(cmds))


for t in table_list:
    if t == 'date':
        table_name = 'ddate'
    else:
        table_name = t
    
    bkt_del_cmd = f'/opt/couchbase/bin/couchbase-cli bucket-delete -c 127.0.0.1 -u couchbase -p couchbase --bucket tpch_{table_name}'
    bkt_create_cmd = f'/opt/couchbase/bin/couchbase-cli bucket-create -c 127.0.0.1 -u couchbase -p couchbase --bucket tpch_{table_name} --bucket-type couchbase --bucket-ramsize 1000'
    import_cmd = f"/opt/couchbase/bin/cbimport csv -c 127.0.0.1 -u couchbase -p couchbase -b tpch_{table_name} -d file:///mnt/interpretable-cost-model/datasets/tpch/tables/{table_name}_couchbase.tbl --infer-types -g key::%{table_attrs[t][0]}%"
    

    # print("del buckt cmd: ", bkt_del_cmd)
    # os.system(bkt_del_cmd)
    # print("create buckt cmd: ", bkt_create_cmd)
    # os.system(bkt_create_cmd)
    # print("import cmd: ", import_cmd)
    # os.system(import_cmd)

    # # Print index
    # print(f'CREATE PRIMARY INDEX tpch_{table_name}_primary ON `tpch_{table_name}`;')
    # print(f'CREATE INDEX tpch_{table_name}_all ON `tpch_{table_name}` ( {",".join(table_attrs[t])} );')
    
    # print(f'DROP INDEX `tpch_{table_name}`.tpch_{table_name}_all;')
    
    print(f'CREATE INDEX tpch_{table_name}_key ON `tpch_{table_name}` ( {table_indexed_keys[t]} );')
    
    # for attr in table_attrs[t]:
    #     print(f'DROP INDEX tpch_{table_name}.tpch_{table_name}_{attr};')
    # for attr in table_attrs[t]:
    #     print(f'CREATE INDEX tpch_{table_name}_{attr} ON `tpch_{table_name}` ( {attr} );')
        
    # print(" ", f"UPDATE STATISTICS FOR `tpch_{table_name}` ({','.join(table_attrs[t])});")
