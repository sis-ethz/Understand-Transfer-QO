
import os
from os import path

table_list = ['movie_info_idx', 'movie_info',
                       'movie_companies', 'movie_keyword', 'cast_info', 'title']

table_attrs = {
    'movie_info_idx': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'movie_info': ['id', 'movie_id', 'info_type_id', 'info', 'note'],
    'movie_companies': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
    'movie_keyword': ['id', 'movie_id', 'keyword_id'],
    'cast_info': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
    'title': ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum']
}

table_indexed_keys = {
    'movie_info_idx': 'id',
    'movie_info': 'id',
    'movie_companies': 'id',
    'movie_keyword': 'id',
    'cast_info': 'id',
    'title': 'id'
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


def create_index_on_each_attr(table_list, table_attrs):
    cmds = []

    create_index_cmd = f'/opt/couchbase/bin/cbq -c 127.0.0.1 -u couchbase -p couchbase '

    for t in table_list:
        if t == 'date':
            table_name = 'ddate'
        else:
            table_name = t

        for attr in table_attrs[t]:
            cmds.append(f'CREATE INDEX imdb_{table_name}_{attr} ON `imdb_{table_name}` ( {attr} );')
    
    print('\n'.join(cmds))



for t in table_list:
    if t == 'date':
        table_name = 'ddate'
    else:
        table_name = t
    
    bkt_del_cmd = f'/opt/couchbase/bin/couchbase-cli bucket-delete -c 127.0.0.1 -u couchbase -p couchbase --bucket imdb_{table_name}'
    bkt_create_cmd = f'/opt/couchbase/bin/couchbase-cli bucket-create -c 127.0.0.1 -u couchbase -p couchbase --bucket imdb_{table_name} --bucket-type couchbase --bucket-ramsize 10000'
    import_cmd = f"/opt/couchbase/bin/cbimport csv -c 127.0.0.1 -u couchbase -p couchbase -b imdb_{table_name} -d file:///mnt/interpretable-cost-model/datasets/imdb_old/tables/{table_name}_job_pdfixed.csv --infer-types -g key::%{table_attrs[t][0]}%"
    create_index_cmd = f''

    # print("del buckt cmd: ", bkt_del_cmd)
    # os.system(bkt_del_cmd)
    # print("create buckt cmd: ", bkt_create_cmd)
    # os.system(bkt_create_cmd)
    # print("import cmd: ", import_cmd)
    # os.system(import_cmd)

    # # Print index
    # print(f'CREATE PRIMARY INDEX imdb_{table_name}_primary ON `imdb_{table_name}`;')
    # print(f'CREATE INDEX imdb_{table_name}_all ON `imdb_{table_name}` ( {",".join(table_attrs[t])} );')
    
    print(f'CREATE INDEX imdb_{table_name}_key ON `imdb_{table_name}` ( {table_indexed_keys[t]} );')

    # for attr in table_attrs[t]:
        # print(f"DROP INDEX `ssb_{table_name}`.ssb_{table_name}_{attr};")
        # print(f'CREATE INDEX imdb_{table_name}_{attr} ON `imdb_{table_name}` ( {attr} );')

    # print(" ", f"UPDATE STATISTICS FOR `imdb_{table_name}` ({','.join(table_attrs[t])});")
